import { Field, ZkProgram, Int64, Provable, verify } from 'o1js';
import * as fs from 'fs/promises';
import { relu } from './relu';

// 선형 변환을 수행하는 모듈화된 레이어 함수
function linearLayer(input: Int64[], weights: Int64[], bias: Int64): Int64 {
  let z = Int64.from(0);
  for (let i = 0; i < weights.length; i++) {
    z = z.add(weights[i].mul(input[i]));
  }
  z = z.add(bias);
  return z;
}

// MLP 모델 정의
const MLP = ZkProgram({
  name: 'MLP',
  publicOutput: Int64,
  methods: {
    predict: {
      privateInputs: [Provable.Array(Int64, 5)], // 5개의 입력값
      async method(input: Int64[]): Promise<Int64> {

        // 첫 번째 히든 레이어
        const weights1: Int64[] = [Int64.from(2), Int64.from(4), Int64.from(3), Int64.from(1), Int64.from(5)];
        const bias1: Int64 = Int64.from(3);
        const z1 = linearLayer(input, weights1, bias1);
        const a1 = relu(z1);

        // 두 번째 히든 레이어
        const weights2: Int64[] = [Int64.from(3), Int64.from(1), Int64.from(4), Int64.from(2), Int64.from(6)];
        const bias2: Int64 = Int64.from(2);
        const z2 = linearLayer([a1, a1, a1, a1, a1], weights2, bias2); // 각 z1 값을 복제해서 전달
        const a2 = relu(z2);

        // 출력 레이어
        const weights3: Int64[] = [Int64.from(1)]; // 활성화 함수 출력값 하나에 대한 가중치
        const bias3: Int64 = Int64.from(5);
        const z3 = linearLayer([a2], weights3, bias3);

        // 최종 출력값 반환
        return z3;
      },
    },
  },
});

(async () => {
  console.log("start");

  // 입력 데이터 (5개의 입력값)
  let input = [Int64.from(25), Int64.from(15), Int64.from(10), Int64.from(5), Int64.from(3)];

  // 증명 키 컴파일
  const { verificationKey } = await MLP.compile();
  console.log('making proof');

  // 예측 수행
  const proof = await MLP.predict(input);
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

