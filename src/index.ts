
import { Field, ZkProgram, Int64, Provable, verify } from 'o1js';

// 선형 회귀 모델 정의
const LinearRegression = ZkProgram({
  name: 'LinearRegression',
  publicOutput: Int64,
  methods: {
    predict: {
      privateInputs: [Provable.Array(Int64, 2)],
      async method(input: Int64[]): Promise<Int64> {
        const coefficients = [Int64.from(5), Int64.from(5)]; // 예제용 계수
        const intercept = Int64.from(0); // 예제용 절편
        let dotProduct = Int64.from(0);

        for (let i = 0; i < coefficients.length; i++) {
          dotProduct = dotProduct.add(coefficients[i].mul(input[i]));
        }

        const z = dotProduct.div(10).add(intercept);
        return z;
      },
    },
  },
});

(async () => {
  console.log("start");

  // 입력 데이터
  let input = [Int64.from(25), Int64.from(35)];

  // 증명 키 컴파일
  const { verificationKey } = await LinearRegression.compile();
  console.log('making proof');

  // 증명 생성
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

