"""Competitor harnesses: the code that trains somebody else's method on our data.

A published scalar and a number we measured are not the same kind of evidence, and this
package exists to produce the second kind. Citing "GLAD reports 0.80" places our result
beside theirs on trust: their split, their thresholds, their AP definition, their hardware.
Training the competitor ourselves, on our splits, scored by our evaluator, gives a PAIRED
measurement -- the same sequences, the same seeds, the same metric -- which is the only
thing a significance test can be run over.

Both belong in the results tables, labelled as what they are. Neither substitutes for the
other: the published scalar is the check that our harness reproduces the method roughly as
its authors got it, and the paired run is the comparison we can actually defend.

Rules for anything added here:

1. Reproduce the method, do not improve it. Every deviation from upstream is enumerated in
   the module that makes it. Beating a competitor we quietly weakened is worse than losing.
2. Give it its published recipe, not ours. If it trains for 100 epochs at 1280 px, it gets
   100 epochs at 1280 px, even when ours gets 30 at 640. A win bought by starving the
   baseline is not a win, and the compute is cheaper than the retraction.
3. Vendor what is small and self-contained; the numbers must not move because an upstream
   repository did.
"""
