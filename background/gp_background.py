import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(["fivethirtyeight", "seaborn-whitegrid", "seaborn-ticks"])
from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.ticker as plticker

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["figure.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

import seaborn.apionly as sns

colors = ["windows blue", "amber", "faded green", "dusty purple"]
colours = sns.xkcd_palette(colors)

import numpy as np
import os
import re
import argparse

from pprint import pprint
from tqdm import tqdm

import pandas as pd

from scipy.stats import binned_statistic_2d, binned_statistic

import torch
import gpytorch

import uproot


def plotHistogram(
    hist,
    xName="",
    yName="",
    zName="",
    plotRange=((0, 1), (0, 1)),
    fileName="hist2d",
    removeZero=True,
):

    hist = hist.copy()
    if removeZero:
        hist[hist < 1e-9] = np.nan

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(111)

    ax1.tick_params(direction="in", pad=10)

    im = plt.imshow(
        hist.T,
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
        extent=(plotRange[0][0], plotRange[0][1], plotRange[1][0], plotRange[1][1]),
        aspect="auto",
    )

    plt.xlabel(xName, fontsize=36)
    plt.ylabel(yName, fontsize=36)

    plt.tick_params(axis="x", colors="k")
    plt.tick_params(axis="y", colors="k")
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)

    cbar_ax = fig.add_axes([1.0, 0.15, 0.04, 0.80])
    cbar = fig.colorbar(im, cax=cbar_ax)
    plt.yticks(fontsize=28)
    cbar_ax.tick_params(direction="in", pad=10)
    plt.ylabel(zName, fontsize=36, color="k", labelpad=20)

    plt.savefig(fileName + ".pdf", dpi=300, bbox_inches="tight")
    plt.clf()


def normalise(hist, totalYield, bins, rescale=False):

    newHist = hist.copy()
    totalHist = np.sum(hist)
    binSizes = bins[1:] - bins[:-1]
    binSum = np.sum(binSizes)
    binFracs = binSizes / (binSum)
    binFracs /= np.min(binFracs)

    for i, b in enumerate(newHist):
        s = np.sum(b)
        c = totalYield / totalHist
        b *= c
        if rescale:
            b /= binFracs[i]

    return newHist


def fitGP(xy, data, mass):

    data = data.reshape(-1)

    xy = torch.tensor(xy).float()
    data = torch.tensor(data).float()

    print(xy.shape)
    print(data.shape)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=3)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    if torch.cuda.is_available():
        xy = xy.cuda()
        data = data.cuda()

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(xy, data, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iter = 1000

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}],  # Includes GaussianLikelihood parameters
        lr=0.05,
    )

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(xy)
        # Calc loss and backprop gradients
        loss = -mll(output, data)
        loss.backward()
        if i % 50 == 0:
            print("Iter %d/%d - Loss: %.3f " % (i, training_iter, loss.item()))
        optimizer.step()

    plotGPProj(model, likelihood, mass)


def plotGPProj(model, likelihood, mass):

    nBinsGP = 100
    nBinsHist = 50

    binsXGP = np.linspace(0.0, 1.0, nBinsGP + 1)
    binsYGP = np.linspace(0.0, 1.0, nBinsGP + 1)

    binCentresXGP = (binsXGP + 0.5 * (binsXGP[1] - binsXGP[0]))[:-1]
    binCentresYGP = (binsYGP + 0.5 * (binsYGP[1] - binsYGP[0]))[:-1]

    binsXHist = np.linspace(0.0, 1.0, nBinsHist + 1)
    binsYHist = np.linspace(0.0, 1.0, nBinsHist + 1)

    binCentresXHist = (binsXHist + 0.5 * (binsXHist[1] - binsXHist[0]))[:-1]
    binCentresYHist = (binsYHist + 0.5 * (binsYHist[1] - binsYHist[0]))[:-1]

    dp = np.zeros((nBinsGP, nBinsGP))

    model.eval()
    likelihood.eval()

    # v. slow not to do it as a single batch
    for i, xi in enumerate(tqdm(binCentresXGP)):
        for j, xj in enumerate(binCentresYGP):

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                v = torch.tensor(np.array([mass, xi, xj]).reshape(-1, 3)).float()
                if torch.cuda.is_available():
                    v = v.cuda()
                observed_pred = likelihood(model(v))

            dp[j][i] = observed_pred.mean.cpu().numpy()  # .reshape(nBinsGP, nBinsGP)

    plotHistogram(dp)  # @ D mass

    predMp = np.sum(dp, 0)
    predThp = np.sum(dp, 1)

    # dataSignal = data.query('md > ' + str(mD - 0.05) + ' and md < ' + str(mD + 0.05) )[['mprime', 'thetaprime']].values
    # massHist, bx, by = np.histogram2d(dataSignal[:,0], dataSignal[:,1], bins = (binsXHist, binsYHist))

    # dataMp = np.sum(massHist, 1)
    # dataThp = np.sum(massHist, 0)

    # predMp = predMp * (np.sum(dataMp) / np.sum(predMp)) * (nBinsGP / nBinsHist)
    # predThp = predThp * (np.sum(dataThp) / np.sum(predThp)) * (nBinsGP / nBinsHist)

    plt.plot(binCentresXGP, predMp)
    # plt.plot(binCentresXHist, dataMp, '.')
    plt.savefig("testBkgMp.pdf")
    plt.clf()

    plt.plot(binCentresYGP, predThp)
    # plt.plot(binCentresYHist, dataThp, '.')
    plt.savefig("testBkgThp.pdf")
    plt.clf()


def makeHistogram(d, binsLower, binsUpper, nBinsDP):

    binsM = binsMLower + binsMUpper

    binsMp = np.linspace(0.0, 1.0, nBins + 1)
    binsThp = np.linspace(0.0, 1.0, nBins + 1)

    hLower = np.histogramdd(d, (binsMLower, binsMp, binsThp))
    hUpper = np.histogramdd(d, (binsMUpper, binsMp, binsThp))

    binCentresLower = [
        (hLower[1][0][:-1] + hLower[1][0][1:]) / 2.0,
        (hLower[1][1][:-1] + hLower[1][1][1:]) / 2.0,
        (hLower[1][2][:-1] + hLower[1][2][1:]) / 2.0,
    ]
    binCentresUpper = [
        (hUpper[1][0][:-1] + hUpper[1][0][1:]) / 2.0,
        (hUpper[1][1][:-1] + hUpper[1][1][1:]) / 2.0,
        (hUpper[1][2][:-1] + hUpper[1][2][1:]) / 2.0,
    ]

    h = np.concatenate((hLower[0], hUpper[0]))

    binCentres = [
        np.concatenate((binCentresLower[0], binCentresUpper[0])),
        binCentresLower[1],
        binCentresLower[2],
    ]

    # normalise yield in bins
    # h = normalise(h, np.sum(h), hLower[1][0], rescale = True)

    newHist = h
    x = []
    y = []

    for i in range(len(binsM) - 2):
        for j in range(len(binsMp) - 1):
            for k in range(len(binsThp) - 1):
                x.append([binCentres[0][i], binCentres[1][j], binCentres[2][k]])
                y.append(newHist[i][j][k])

    y = np.array(y).reshape(-1, 1)
    x = np.array(x)

    return x, y


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-f",
        "--inputFile",
        type=str,
        dest="inputFile",
        default="test.root",
        help="Input file name.",
    )
    argParser.add_argument(
        "-t", "--tree", type=str, dest="treeName", default="tree", help="Tree name."
    )

    argParser.add_argument(
        "-mv", type=str, dest="massVariable", default="md", help="Mass variable name."
    )
    argParser.add_argument(
        "-x", type=str, dest="x", default="mprime", help="x variable name."
    )
    argParser.add_argument(
        "-y", type=str, dest="y", default="thetaprime", help="y variable name."
    )

    argParser.add_argument(
        "-m", type=float, dest="mass", default=2.045, help="Mass value to eval GP at."
    )

    argParser.add_argument(
        "-n", "--nXY", type=int, dest="nBins", default=25, help="Number of xy bins."
    )

    argParser.add_argument(
        "-c",
        "--cut",
        type=str,
        dest="cutString",
        default="md > 1.77 and md < 2.17",
        help="Cut string (to remove signal).",
    )

    argParser.add_argument(
        "-l",
        "--lowerBins",
        type=float,
        dest="lowerBins",
        nargs="+",
        help="Lower (less than m(X)) bin edges (space separated)",
        required=True,
    )
    argParser.add_argument(
        "-u",
        "--upperBins",
        type=float,
        dest="upperBins",
        nargs="+",
        help="Upper (less than m(X)) bin edges (space separated)",
        required=True,
    )

    args = argParser.parse_args()

    vars = [args.massVariable, args.x, args.y]

    data = uproot.open(args.inputFile)[args.treeName].arrays(library="pd")[vars]
    data = data.query(args.cutString)[vars].to_numpy()

    histX, y = makeHistogram(data, args.lowerBins, args.upperBins, args.nBins)

    fitGP(histX, y)
