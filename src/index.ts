import { Field, ZkProgram, Int64, Provable, verify } from 'o1js';

// 더미 데이터
const dummyData = [
  { Feature: 1, Target: 3 },
  { Feature: 2, Target: 5 },
  { Feature: 3, Target: 7 },
  { Feature: 4, Target: 9 },
  { Feature: 5, Target: 11 },
  { Feature: 6, Target: 13 },
  { Feature: 7, Target: 15 },
  { Feature: 8, Target: 17 },
  { Feature: 9, Target: 19 },
  { Feature: 10, Target: 21 },
];

// 선형 회귀 모델 정의
const LinearRegression = ZkProgram({
  name: 'LinearRegression',
  publicOutput: Int64,
  methods: {
    predict: {
      privateInputs: [Provable.Array(Int64, 1)],
      async method(input: Int64[]): Promise<Int64> {
        // 더미 데이터를 Int64로 변환
        const data = dummyData.map(d => [Int64.from(d.Feature), Int64.from(d.Target)]);

        // X와 y를 분리
        const X = data.map(d => [d[0], Int64.from(1)]); // feature와 상수항 추가
        const y = data.map(d => d[1]); // target

        // 행렬 연산
        const Xt = transposeMatrix(X);
        const XtX = matrixMultiply(Xt, X);
        const Xty = Xt.map(row => row.reduce((acc, val, i) => acc.add(val.mul(y[i])), Int64.from(0)));
        const XtX_inv = inverse2x2Matrix(XtX);
        const coefficients = XtX_inv.map(row => row.reduce((acc, val, i) => acc.add(val.mul(Xty[i])), Int64.from(0)));

        // 예측 수행
        let dotProduct = Int64.from(0);
        for (let i = 0; i < coefficients.length - 1; i++) {
          dotProduct = dotProduct.add(coefficients[i].mul(input[i]));
        }

        const intercept = coefficients[coefficients.length - 1];
        const z = dotProduct.add(intercept);
        return z;
      },
    },
  },
});

// 행렬 전치 함수
function transposeMatrix(A: Int64[][]): Int64[][] {
  const rows = A.length;
  const cols = A[0].length;
  let T = Array(cols).fill(null).map(() => Array(rows).fill(Int64.from(0)));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

// 행렬 곱셈 함수
function matrixMultiply(A: Int64[][], B: Int64[][]): Int64[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  let C = Array(rowsA).fill(null).map(() => Array(colsB).fill(Int64.from(0)));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        C[i][j] = C[i][j].add(A[i][k].mul(B[k][j]));
      }
    }
  }
  return C;
}

// 2x2 행렬의 역행렬 함수
function inverse2x2Matrix(A: Int64[][]): Int64[][] {
  const a = A[0][0];
  const b = A[0][1];
  const c = A[1][0];
  const d = A[1][1];

  const det = a.mul(d).sub(b.mul(c));
  // const invDet = det.inv();

  return [
    [d.div(det), b.negV2().div(det)],
    [c.negV2().div(det), a.div(det)]
  ];
}

(async () => {
  console.log("start");

  // 입력 데이터
  let input = [Int64.from(25)];

  // 증명 키 컴파일
  const { verificationKey } = await LinearRegression.compile();
  console.log('making proof');

  // 예측 수행
  const proof = await LinearRegression.predict(input);
  console.log('proof created');

  // 증명 검증 함수
  const verifyProof = async (proof: any, verificationKey: any) => {
    return await verify(proof, verificationKey);
  };

  // 검증 수행
  const isValid = await verifyProof(proof, verificationKey);
  console.log('Proof is valid:', isValid);
})();

