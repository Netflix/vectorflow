/**
 * Utility functions for ML-related tasks.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.ml;

private{
import std.algorithm : map, reverse, sort;
import std.array;
import std.conv : to;
import std.range : evenChunks;
import std.range.primitives;
import std.traits;
import std.typecons : Tuple, tuple;
}


/**
* Compute ROC curve and AUC. Returns the AUC
*
* Params:
*    data = forward range of (true label, prediction)
*    roc_curve = will populate the ROC curve in this array
*    num_cutoff = number of points in the ROC curve
*/
double computeROCandAUC(R)(R data, out Tuple!(float, float)[] roc_curve,
                           ulong num_cutoff = 400)
    if(isForwardRange!R)
{
    // sort by predictions
    auto sorted = data.sort!((a, b) => a[1] < b[1]).map!(x => x[0]);
    float all_pos = 0;
    float all_neg = 0;
    foreach(l; sorted)
    {
        if(l > 0)
            all_pos++;
        else
            all_neg++;
    }

    auto chunks = evenChunks(sorted, num_cutoff);

    roc_curve.length = 0;

    float sum_pos = 0;
    float sum_neg = 0;
    foreach(chunk; chunks)
    {
        foreach(l; chunk)
        {
            if(l > 0)
            {
                sum_pos++;
                all_pos--;
            }
            else
            {
                sum_neg++;
                all_neg--;
            }
        }
        float tpr = all_pos / (all_pos + sum_pos);
        float tnr = sum_neg / (all_neg + sum_neg);
        roc_curve ~= tuple(1.0f - tnr, tpr);
    }
    reverse(roc_curve);

    double auc = 0.0;
    // origin triangle first
    float h = roc_curve[0][1] - 0;
    float w = roc_curve[0][0] - 0;
    auc += h * w / 2;

    // then all other triangles
    foreach(i; 0..roc_curve.length - 1)
    {
        float height = roc_curve[i + 1][1] - roc_curve[i][1];
        float width = roc_curve[i + 1][0] - roc_curve[i][0];
        auc += height * width / 2 + roc_curve[i][1] * width;
    }
    // then endpoint triangle
    h = 1.0 - roc_curve[$-1][1];
    w = 1.0 - roc_curve[$-1][0];
    auc += h * w / 2 + roc_curve[$-1][1] * w;

    return auc;
}


/**
* Compute the histogram of an InputRange using uniform bins
*
* Params:
*    vals = InputRange of numerical values
*    num_bins = number of bins in the histogram returned
*    bin_min = lower bound of the histogram range
*    bin_max = higher bound of the histogram range
*    bins = intervals [a, b[ used for binning
*    normalized = whether or not the histogram is normalized by its sum
*/
float[] histogram(D)(D vals, ushort num_bins, float bin_min, float bin_max,
                     out float[] bins, bool normalized = false)
    if(isInputRange!D && isNumeric!(ElementType!D))
{
    auto hist = new float[num_bins];
    hist[] = 0;

    auto bins_left_bound = new float[num_bins];
    foreach(i; 0..num_bins)
        bins_left_bound[i] = bin_min + (bin_max - bin_min) * i.to!float / (num_bins - 1);
    bins = bins_left_bound;

    foreach(ref v; vals)
    {
        ulong bin_ind = 0;
        while(bin_ind < bins_left_bound.length && v >= bins_left_bound[bin_ind])
            ++bin_ind;
        if(bin_ind < bins_left_bound.length)
            bin_ind--;
        if(bin_ind >= 0 && bin_ind < bins_left_bound.length)
            hist[bin_ind] += 1;
    }
    if(normalized)
        hist[] /= hist.sum();

    return hist;
}

