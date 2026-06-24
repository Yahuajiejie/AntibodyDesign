"""Removed: heap-array pair construction.

Tried as an alternative to `tree.py`'s median-split tree -- reading a
rank-sorted group directly through the implicit binary-heap index scheme
(`child(i) = 2i+1, 2i+2`) instead of recursively picking medians. Measured
the same O(n) edge count and the same O(log n) diameter as the median tree
(see the design discussion this came out of), so it looked like a cheaper
drop-in replacement at first.

It isn't one. Reading heap indices off a sorted array makes the root the
group's single lowest-affinity record, and that choice forces every index
`i >= n // 2` to be a leaf -- the entire upper half of the group's rank
range ends up with only one comparison each, with zero comparisons landing
in the lower half. That's not a tunable knob; it's what "child index is
always larger than parent index" means once the array is sorted. The only
way to remove the bias is to stop indexing the sorted array directly and
instead lay out the median-split tree's shape into the array (so the root
is the median, not the minimum) -- at which point it is no longer a heap
read off sorted order, it is `tree.py`'s tree with a different storage
layout. There's no intermediate version that keeps the heap's "just read
sorted-array indices" simplicity and fixes the bias.

So this strategy was dropped rather than patched. Kept as an empty module
(deletion isn't available in this environment) -- nothing imports from it;
`pair_sample_strategy` no longer accepts `"heap_array"`. See `tree.py` and
`randomized_tree.py` for the two strategies actually in use.
"""

from __future__ import annotations
