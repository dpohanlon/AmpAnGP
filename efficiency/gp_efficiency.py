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


def plotPull(pull, xName, yName, plotRange, fileName):

    print(np.sum(pull) / (50 * 50 - 5))

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(111)

    ax1.tick_params(direction="in", pad=10)

    cmap = plt.cm.RdBu
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(-5, 5, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    im = plt.imshow(
        pull,
        cmap=cmap,
        norm=norm,
        extent=(plotRange[0][0], plotRange[0][1], plotRange[1][0], plotRange[1][1]),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    ax1.tick_params(direction="in", pad=10)
    ax1.tick_params(direction="in", pad=10)

    plt.xlabel(xName, fontsize=36)
    plt.ylabel(yName, fontsize=36)

    plt.tick_params(axis="x", colors="k")
    plt.tick_params(axis="y", colors="k")
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)

    ax2 = fig.add_axes([1.0, 0.15, 0.07, 0.8])
    cb = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds,
        boundaries=bounds,
    )

    plt.yticks(fontsize=28)
    ax2.tick_params(direction="in", pad=10)
    plt.ylabel("Pull", fontsize=36, color="k", labelpad=20)

    plt.savefig(fileName + ".pdf", dpi=300, bbox_inches="tight")
    plt.clf()


def calculatePull(hist1, hist2):
    # Calculate error using hist1

    error = np.sqrt(hist1)

    return (hist1 - hist2) / error


def calculateChiSq(hist1, hist2):

    return (hist2 - hist1) ** 2 / hist2


def makeDPHistogram(
    data, var1, var2, name="dp", binning=(30, 30), plotRange=((0, 1), (0, 1)), plot=True
):

    binsX = np.linspace(plotRange[0][0], plotRange[0][1], binning[0] + 1)
    binsY = np.linspace(plotRange[1][0], plotRange[1][1], binning[1] + 1)

    hist, x, y = np.histogram2d(data[var1], data[var2], bins=(binsX, binsY))

    if plot:
        plotHistogram(
            hist,
            r"$m'$",
            r"$\theta'$",
            zName="Efficiency",
            plotRange=plotRange,
            fileName=name,
        )

    return hist, binsX, binsY


def fitGP(data, binsX, binsY, plotRange=((0, 1), (0, 1)), plotBins=1000):

    binCentresX = (binsX + 0.5 * (binsX[1] - binsX[0]))[:-1]
    binCentresY = (binsY + 0.5 * (binsY[1] - binsY[0]))[:-1]

    bx, by = np.meshgrid(binCentresX, binCentresY)
    xy = np.vstack((bx.reshape(-1), by.reshape(-1)))
    xy = xy.T

    data = data.reshape(-1)

    xy = torch.tensor(xy).float()
    data = torch.tensor(data).float()

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=2)
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

    training_iter = 100

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}],  # Includes GaussianLikelihood parameters
        lr=0.01,
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

    plotXs = np.linspace(plotRange[0][0], plotRange[0][1], plotBins)
    plotYs = np.linspace(plotRange[1][0], plotRange[1][1], plotBins)

    bxp, byp = np.meshgrid(plotXs, plotYs)
    xyp = np.vstack((bxp.reshape(-1), byp.reshape(-1)))
    xyp = xyp.T

    xyp = torch.tensor(xyp).float()

    if torch.cuda.is_available():
        xyp = xyp.cuda()
        data = data.cpu()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(xyp))

    gpPred = observed_pred.mean.cpu().numpy().reshape(plotBins, plotBins)

    plotHistogram(
        (gpPred / np.amax(data.numpy())),
        r"$m'$",
        r"$\theta'$",
        "Efficiency",
        plotRange=plotRange,
        fileName="gpFit_DsKpipi",
        removeZero=False,
    )

    return gpPred


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
        "-x", type=str, dest="x", default="mprime", help="x variable name."
    )
    argParser.add_argument(
        "-y", type=str, dest="y", default="thetaprime", help="y variable name."
    )

    argParser.add_argument(
        "-n", "--nBins", type=int, dest="nBins", default=25, help="Number of xy bins."
    )

    args = argParser.parse_args()

    data = uproot.open(args.inputFile)[args.treeName].arrays(library="pd")[
        [args.x, args.y]
    ]

    hist, binsX, binsY = makeDPHistogram(
        data1, args.x, args.y, binning=(args.nBins, args.nBins), plot=True
    )

    gpHist = fitGP(hist, binsX, binsY, plotBins=100)

    # Add pull (etc) calculation here
