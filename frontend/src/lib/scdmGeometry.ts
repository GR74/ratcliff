/** Display-only affine mapping; use all samples, never absolute evidence. */
export function evidenceRadius(evidence: number[][], threshold: number) {
  let min = 0, max = threshold;
  for (const row of evidence) for (const value of row) {
    min = Math.min(min, value); max = Math.max(max, value);
  }
  const scale = 1 / Math.max(max - min, 1);
  const r0 = 1 - scale * min;
  // The most negative observed evidence has radius 1; all angles stay fixed.
  return { min, max, scale, r0, radius: (value: number) => r0 + scale * value };
}

/** Split straight display edges at zero; this does not interpolate model readout. */
export function signedSegments(values: number[], times: number[]) {
  const positive: (number[] | null)[] = [], negative: (number[] | null)[] = [];
  for (let i = 1; i < values.length; i++) {
    const a = [values[i - 1], times[i - 1]], b = [values[i], times[i]];
    if (a[0] * b[0] < 0) {
      const f = -a[0] / (b[0] - a[0]);
      const zero = [0, a[1] + f * (b[1] - a[1])];
      (a[0] < 0 ? negative : positive).push(a, zero, null);
      (b[0] < 0 ? negative : positive).push(zero, b, null);
    } else (a[0] < 0 || b[0] < 0 ? negative : positive).push(a, b, null);
  }
  return { positive, negative };
}
