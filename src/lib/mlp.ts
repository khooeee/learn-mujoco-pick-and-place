/** 6→64→64→3 MLP: object pose + size + shape → grasp point. */

export type MLP = {
  inSize: number;
  h1: number;
  h2: number;
  outSize: number;
  W1: number[];
  b1: number[];
  W2: number[];
  b2: number[];
  W3: number[];
  b3: number[];
};

function randn() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function fill(n: number, scale: number) {
  return Array.from({ length: n }, () => randn() * scale);
}

export function initMLP(inSize = 6, h1 = 64, h2 = 64, outSize = 3): MLP {
  return {
    inSize,
    h1,
    h2,
    outSize,
    W1: fill(h1 * inSize, Math.sqrt(2 / inSize)),
    b1: Array(h1).fill(0),
    W2: fill(h2 * h1, Math.sqrt(2 / h1)),
    b2: Array(h2).fill(0),
    W3: fill(outSize * h2, Math.sqrt(2 / h2)),
    b3: Array(outSize).fill(0),
  };
}

export function paramCount(m: MLP) {
  return m.W1.length + m.b1.length + m.W2.length + m.b2.length + m.W3.length + m.b3.length;
}

function tanh(x: number) {
  return Math.tanh(x);
}

export function forward(m: MLP, x: number[]): number[] {
  const a1 = new Array(m.h1);
  for (let i = 0; i < m.h1; i++) {
    let s = m.b1[i];
    for (let j = 0; j < m.inSize; j++) s += m.W1[i * m.inSize + j] * x[j];
    a1[i] = tanh(s);
  }
  const a2 = new Array(m.h2);
  for (let i = 0; i < m.h2; i++) {
    let s = m.b2[i];
    for (let j = 0; j < m.h1; j++) s += m.W2[i * m.h1 + j] * a1[j];
    a2[i] = tanh(s);
  }
  const y = new Array(m.outSize);
  for (let i = 0; i < m.outSize; i++) {
    let s = m.b3[i];
    for (let j = 0; j < m.h2; j++) s += m.W3[i * m.h2 + j] * a2[j];
    y[i] = s;
  }
  return y;
}

export function trainEpoch(m: MLP, xs: number[][], ys: number[][], lr: number): number {
  const n = xs.length;
  let loss = 0;
  const gW1 = new Array(m.W1.length).fill(0);
  const gb1 = new Array(m.h1).fill(0);
  const gW2 = new Array(m.W2.length).fill(0);
  const gb2 = new Array(m.h2).fill(0);
  const gW3 = new Array(m.W3.length).fill(0);
  const gb3 = new Array(m.outSize).fill(0);

  for (let k = 0; k < n; k++) {
    const x = xs[k];
    const t = ys[k];
    const a1 = new Array(m.h1);
    for (let i = 0; i < m.h1; i++) {
      let s = m.b1[i];
      for (let j = 0; j < m.inSize; j++) s += m.W1[i * m.inSize + j] * x[j];
      a1[i] = tanh(s);
    }
    const a2 = new Array(m.h2);
    for (let i = 0; i < m.h2; i++) {
      let s = m.b2[i];
      for (let j = 0; j < m.h1; j++) s += m.W2[i * m.h1 + j] * a1[j];
      a2[i] = tanh(s);
    }
    const y = new Array(m.outSize);
    for (let i = 0; i < m.outSize; i++) {
      let s = m.b3[i];
      for (let j = 0; j < m.h2; j++) s += m.W3[i * m.h2 + j] * a2[j];
      y[i] = s;
    }

    const dy = new Array(m.outSize);
    for (let i = 0; i < m.outSize; i++) {
      dy[i] = (2 / m.outSize) * (y[i] - t[i]);
      loss += (y[i] - t[i]) ** 2;
    }

    const da2 = new Array(m.h2).fill(0);
    for (let i = 0; i < m.outSize; i++) {
      gb3[i] += dy[i];
      for (let j = 0; j < m.h2; j++) {
        gW3[i * m.h2 + j] += dy[i] * a2[j];
        da2[j] += m.W3[i * m.h2 + j] * dy[i];
      }
    }
    const dz2 = da2.map((v, i) => (1 - a2[i] * a2[i]) * v);
    const da1 = new Array(m.h1).fill(0);
    for (let i = 0; i < m.h2; i++) {
      gb2[i] += dz2[i];
      for (let j = 0; j < m.h1; j++) {
        gW2[i * m.h1 + j] += dz2[i] * a1[j];
        da1[j] += m.W2[i * m.h1 + j] * dz2[i];
      }
    }
    const dz1 = da1.map((v, i) => (1 - a1[i] * a1[i]) * v);
    for (let i = 0; i < m.h1; i++) {
      gb1[i] += dz1[i];
      for (let j = 0; j < m.inSize; j++) gW1[i * m.inSize + j] += dz1[i] * x[j];
    }
  }

  const inv = 1 / n;
  loss = loss / (n * m.outSize);
  const step = (arr: number[], g: number[]) => {
    for (let i = 0; i < arr.length; i++) arr[i] -= lr * g[i] * inv;
  };
  step(m.W1, gW1);
  step(m.b1, gb1);
  step(m.W2, gW2);
  step(m.b2, gb2);
  step(m.W3, gW3);
  step(m.b3, gb3);
  return loss;
}
