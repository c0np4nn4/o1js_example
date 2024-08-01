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

        // Xt와 Xty 계산
        const Xt = transposeMatrix(X);
        const Xty = Xt.map(row => row.reduce((acc, val, i) => acc.add(val.mul(y[i])), Int64.from(0)));

        // LU 분해 또는 가우스 소거법 등을 사용하여 정수 연산 기반 계수 추정
        // 여기서는 단순 예제를 위해서 OLS 공식 사용을 가정하고 정수 비율 유지
        // 아래 코드는 실제 정수 계산으로 작성되어야 함 (예: 교차검증에서 사용)

        // XtX와 XtX_inv를 사용하지 않고 직접적으로 해결
        let beta0 = Int64.from(0); // 절편
        let beta1 = Int64.from(0); // 기울기

        // 예제: 직접적으로 회귀 계수를 계산하지 않고, 정수 계산 예시
        const n = Int64.from(globalData.length);
        const sumX = globalData.reduce((acc, d) => acc.add(d[0]), Int64.from(0));
        const sumY = y.reduce((acc, d) => acc.add(d), Int64.from(0));
        const sumXY = globalData.reduce((acc, d) => acc.add(d[0].mul(d[1])), Int64.from(0));
        const sumXX = globalData.reduce((acc, d) => acc.add(d[0].mul(d[0])), Int64.from(0));

        beta1 = (n.mul(sumXY).sub(sumX.mul(sumY))).div(n.mul(sumXX).sub(sumX.mul(sumX)));
        beta0 = (sumY.sub(beta1.mul(sumX))).div(n);

        // 예측 수행
        let prediction = beta0.add(beta1.mul(input[0]));

        return prediction;
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

(async () => {
  console.log("start");

  // JSON 데이터 로드
  await loadData('data.json');

  // 입력 데이터
  let input = [Int64.from(70)];

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

