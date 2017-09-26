/**
 * Internal abstraction to report progress during training.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.monitor;

private
{
import core.time : dur, Duration, MonoTime, ticksToNSecs;
import std.algorithm : max, filter, sum;
import std.array;
import std.conv : to;
import std.format : format, sformat;
import std.math : lround;
import std.stdio : stdout;

import vectorflow.math : fabs;
import vectorflow.utils : isTerminal;
}



class SGDMonitor {

    bool verbose;
    ulong num_epochs;
    uint num_cores;
    bool with_loss;
    MonoTime start_time;
    MonoTime last_time;

    ulong[] examples_seen;
    ulong[] features_seen;
    ushort[] passes_seen;
    double[] acc_loss;

    char[] _bar_buff;

    char[] _buff_stdout_line;

    string _pattern;
    bool _isTerminal;
    ushort _max_rows_no_term;
    float _last_percent_displayed;
    ushort _rows_no_term;

    this(bool verbose_, ulong num_epochs_,
            uint num_cores_, MonoTime start_time_, bool with_loss_)
    {
        verbose = verbose_;
        num_epochs = num_epochs_;
        num_cores = num_cores_;
        start_time = start_time_;
        with_loss = with_loss_;
        examples_seen.length = num_cores;
        features_seen[] = 0;
        features_seen.length = num_cores;
        features_seen[] = 0;
        passes_seen.length = num_cores;
        passes_seen[] = 0;
        acc_loss.length = num_cores;
        acc_loss[] = 0.0;

        _bar_buff = new char[100];
        _bar_buff[0] = '[';
        _bar_buff[51] = ']';

        _buff_stdout_line = new char[240];

        if(with_loss_)
            _pattern = 
            "Progress: %s | Elapsed: %s | Remaining: %s | %04d passes " ~
            "| Loss: %.4e | %.2e obs/sec | %.2e features/sec";
        else
            _pattern =
            "Progress: %s | Elapsed: %s | Remaining: %s | %04d passes " ~
            "| %.2e obs/sec | %.2e features/sec";

        _isTerminal = isTerminal();
        if(_isTerminal)
            _pattern ~= "\r";
        else
        {
            _pattern ~= "\n";
            _max_rows_no_term = 20;
            _rows_no_term = 0;
        }
        _last_percent_displayed = 0.0f;
    }

    private char[] get_progress_bar(float percentage)
    {
        _bar_buff[1..51] = ' ';
        auto num_finished = lround(percentage * 50).to!size_t;
        if(num_finished >= 1)
            _bar_buff[1..num_finished+1] = 'o';
        auto end = sformat(_bar_buff[52..100],
                " (%.1f %%)", percentage * 100).length;
        return _bar_buff[0..52+end];
    }

    private static string time_clock_str(Duration d, bool with_ms)
    {
        auto ds = d.split!("hours", "minutes", "seconds", "msecs")();
        if(with_ms)
            return "%02d:%02d:%02d.%03d".format(
                    ds.hours, ds.minutes, ds.seconds, ds.msecs);
        return "%02d:%02d:%02d".format(
                    ds.hours, ds.minutes, ds.seconds);
    }

    void progress_callback(uint core_id, ulong epoch, ulong num_examples,
            ulong num_features, double sum_loss)
    {
        if(!verbose)
            return;
        last_time = MonoTime.currTime;
        examples_seen[core_id] += num_examples;
        features_seen[core_id] += num_features;
        acc_loss[core_id] += sum_loss;
        passes_seen[core_id] = epoch.to!ushort;
        auto ticks = (last_time.ticks - start_time.ticks);
        double seconds = ticksToNSecs(ticks).to!double / 1e9;
        auto time = time_clock_str(last_time - start_time, true);

        auto passes = passes_seen.filter!(x => x > 0).array;
        auto avg_passes = passes.sum.to!float / max(1, passes.length);
        auto total_ex_seen = examples_seen.sum.to!double;
        auto total_feats_seen = features_seen.sum.to!double;
        auto total_loss = acc_loss.sum;
        auto avg_loss_per_ex = total_loss / total_ex_seen;

        auto percent = avg_passes/ num_epochs;
        bool write_line = true;
        if(!_isTerminal)
        {
            if(fabs(percent - 1) < 1e-8
                    || percent > _rows_no_term.to!float / _max_rows_no_term)
                _rows_no_term++;
            else
                write_line = false;
        }

        if(!write_line)
            return;
        string predict_remaining = "--------";
        if(percent > 0 && percent < 1)
        {
            auto remaining_secs = seconds / percent - seconds;
            auto remaining_dur = dur!"seconds"(lround(remaining_secs));
            predict_remaining = time_clock_str(remaining_dur, false);
        }
        char[] line;
        auto bar = get_progress_bar(percent);
        if(with_loss)
        {
            line = sformat(_buff_stdout_line, _pattern,
                bar, time, predict_remaining, lround(avg_passes),
                avg_loss_per_ex,
                total_ex_seen / seconds, total_feats_seen / seconds);
        }
        else
        {
            line = sformat(_buff_stdout_line, _pattern,
                bar, time, predict_remaining, lround(avg_passes),
                total_ex_seen / seconds, total_feats_seen / seconds);
        }
        stdout.write(line);
        stdout.flush();
    }

    void wrap_up()
    {
        if(!verbose)
            return;
        if(_isTerminal && num_epochs > 0)
        {
            stdout.write("\n");
            stdout.flush();
        }
    }
}
