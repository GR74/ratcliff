# Draft note to Roger Ratcliff and Aniket: JAX port fidelity, subject 1

*Draft — not sent.*

---

Roger, Aniket,

We checked the JAX port of the 2D model against Roger's TACC fit of subject 1
(`twod3fitmpi.tacc.out`). The port now reproduces the final fit, but only after
we fixed three things it did differently from the Fortran. We'd like one check
from you.

**Method.** We took the start point and the six checkpoints from the log: the best
vertex after each simplex restart, with the G² the Fortran printed for it. We
rescored each with the port, using the same data file (`twod3datanew`), the same
trial count (9,984 per condition, as on 78 × 64 cores) and the exact circulant
(FFT) random field. Each point was scored with five independent random keys
(mean ± s.e.m.). The log's data file, start point and restart schedule are those
of `benchtwod3mpi.f`, so that is the program we compared against.

**Three differences between the port and `benchtwod3mpi.f`.** Line numbers refer to
`benchtwod3mpi.f.new_accum.f.THIS_ONE_WORKS`. Its diff from the May source is MPI
plumbing only.

1. **`sv` was never used.** The Fortran scales each trial's drift by
   `vs = 1 + sv*(gu1(8) − .5)` (line 445). The port read `sv` but never applied it,
   which is why it looked inert in our earlier fits.
2. **G² was computed differently.** The Fortran puts RTs into 10 ms step bins
   (`l2 = rt`, line 532). `qant1` then reads the quantile proportions off that binned
   distribution, interpolating between bins (lines 566 and 575). The port counted
   raw RTs against the quantiles.
3. **Trials that never reached the boundary were categorised differently.** The
   Fortran assigns a trial that never crosses by where its maximum sits at the last
   step (`maxloc`, line 468) and gives it RT step 401. The port put all such trials
   in category 5.

**Results** (G², JAX mean ± s.e.m. over 5 keys):

| Checkpoint | Fortran (log) | Port before fixes | Port after fixes |
|---|---|---|---|
| 1 | 174.7 | 220.1 ± 1.3 | 158.8 ± 0.8 |
| 2 | 167.5 | 198.8 ± 0.9 | 148.2 ± 0.7 |
| 3 | 168.9 | 191.5 ± 1.1 | 146.5 ± 0.9 |
| 4 | 165.9 | 197.5 ± 1.7 | 149.5 ± 1.6 |
| 5 | 152.3 | 201.6 ± 2.2 | 149.1 ± 1.6 |
| 6 (final) | 139.1 | 192.4 ± 1.4 | **139.7 ± 1.7** |

- **Final checkpoint:** after the fixes it matches.
- **Checkpoints 1–4:** the port scores them 16–22 *lower* than the log. We can't
  explain this from the code.
- **Checkpoint order:** in the port, checkpoints 2–5 are nearly tied. In the log
  they spread from 168.9 down to 152.3.
- **K-L field:** with the faster K-L approximation (99% of variance) in place of
  the exact field, every value is 5–8 higher, a bias we will report.

**Our best guess.** The log's G² values might vary much more from one evaluation
to the next than ours do. Each Fortran run seeds from the clock, and the seeding
of individual trials is unusual. If so, the simplex keeps whichever vertex was
evaluated with a lucky seed, and the printed values carry that luck.

**The check we'd like.** Could one of you run `benchtwod3mpi` at a single fixed
parameter set, say the final checkpoint, about 10 times with different clock
seeds and send us the G² values? Their spread would tell us whether the early
gap is evaluation noise or a fourth difference we haven't found.

**A refit of subject 1 under the corrected model.** We fit from your start point
with CMA-ES (3,000 evaluations, about 43 min on one RTX 4090). Both points were
scored the same way (exact field, 9,984 trials, 5 fresh keys):

- our fit: G² 96.9 ± 1.4;
- your final parameters: G² 139.7 ± 1.7.

That holds under the port, which still disagrees with your log at checkpoints
1–4, so we don't yet claim it for your program. The fit had not converged.
Most of the difference is along one direction: all six drift amplitudes and
`sis` are higher (drift 1, condition 1: 20.1 vs 16.2; `sis` 12.8 vs 10.9),
and `st` is lower (36.6 vs 47.7). The drifts and `sis` rose together: drift 1
divided by `sis` barely changed (1.56 vs 1.49). So the fit mostly scaled the
drift amplitudes up while widening the bumps.

**What changes for the earlier work.**
- **All earlier fits are affected:** pass 1 and 2, including Aniket's recovery
  results, were fit with the uncorrected port. The speed-ups still stand.
- **The corrected model is opt-in:** it sits behind a flag
  (`fortran_variant="bench"`). The subject-1 refit above uses it, and we are
  rerunning the optimizer comparison on it.
- **`twod3fitmpi.f`** differs again: smaller category zones, drift-only centring
  and a ter floor of 150. The port supports it too (`fortran_variant="fit"`).
  Which program should be the reference for the other 15 subjects?

Thanks,
Gowri
