import { Field, ZkProgram, Int64, Provable, verify } from 'o1js';
import * as fs from 'fs/promises';

// 전역 데이터 변수
let globalData: Int64[][] = [];

// 데이터 읽기 함수
async function loadData(filePath: string): Promise<void> {
  const data = await fs.readFile(filePath, 'utf8');
  const parsedData = JSON.parse(data);
  globalData = parsedData.map((d: { Feature: number; Target: number }) => [Int64.from(d.Feature), Int64.from(d.Target)]);
}

// 선형 회귀 모델 정의
const LinearRegression = ZkProgram({
  name: 'LinearRegression',
  publicOutput: Int64,
  methods: {
    predict: {
      privateInputs: [Provable.Array(Int64, 1)],
      async method(input: Int64[]): Promise<Int64> {
        // X와 y를 분리
        const X = globalData.map(d => [d[0], Int64.from(1)]); // feature와 상수항 추가
        const y = globalData.map(d => d[1]); // target

        // 행렬 연산
        const Xt = transposeMatrix(X);
        const XtX = matrixMultiply(Xt, X);
        const Xty = Xt.map(row => row.reduce((acc, val, i) => acc.add(val.mul(y[i])), Int64.from(0)));

        // 가우스 소거법으로 회귀 계수 계산
        const coefficients = gaussElimination(XtX, Xty);


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

// 가우스 소거법을 사용하여 선형 시스템을 해결하는 함수
function gaussElimination(A: Int64[][], b: Int64[]): Int64[] {
  const n = A.length;
  let augmentedMatrix: Int64[][] = A.map((row, i) => [...row, b[i]]);

  // Forward elimination
  for (let i = 0; i < n; i++) {
    // Pivot: 절대값이 가장 큰 요소를 찾습니다.
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      // 절대값을 비교하기 위해 음수인지 양수인지 판단
      const currentAbs = augmentedMatrix[k][i] > Int64.from(0)
        ? augmentedMatrix[k][i]
        : augmentedMatrix[k][i].negV2();
      const maxAbs = augmentedMatrix[maxRow][i] > Int64.from(0)
        ? augmentedMatrix[maxRow][i]
        : augmentedMatrix[maxRow][i].negV2();

      if (currentAbs > maxAbs) {
        maxRow = k;
      }
    }

    // Swap maximum row with current row
    [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];

    // Make all rows below this one 0 in current column
    for (let k = i + 1; k < n; k++) {
      const factor = augmentedMatrix[k][i].div(augmentedMatrix[i][i]);
      for (let j = i; j <= n; j++) {
        augmentedMatrix[k][j] = augmentedMatrix[k][j].sub(factor.mul(augmentedMatrix[i][j]));
      }
    }
  }

  // Solve equation Ax = b for an upper triangular matrix A
  let x = Array(n).fill(Int64.from(0));
  for (let i = n - 1; i >= 0; i--) {
    let sum = Int64.from(0);
    for (let j = i + 1; j < n; j++) {
      sum = sum.add(augmentedMatrix[i][j].mul(x[j]));
    }
    x[i] = (augmentedMatrix[i][n].sub(sum)).div(augmentedMatrix[i][i]);
  }
  return x;
}

(async () => {
  console.log("start");

  // JSON 데이터 로드
  await loadData('data.json');

  // 입력 데이터
  let input = [Int64.from(25)];

  // 증명 키 컴파일
  const { verificationKey } = await LinearRegression.compile();
  console.log('making proof');

  // 예측 수행
  const proof = await LinearRegression.predict(input);
  console.log('proof created');
  console.log('value: ', proof.publicOutput.toString());

  // 증명 검증 함수
  const verifyProof = async (proof: any, verificationKey: any) => {
    return await verify(proof, verificationKey);
  };

  // 검증 수행
  const isValid = await verifyProof(proof, verificationKey);
  console.log('Proof is valid:', isValid);
})();

