import jax
import jax.numpy as jnp
import time

def generate_and_multiply():
    size = 2**12
    matrix = jax.random.normal(jax.random.PRNGKey(0), (size, size))
    result = jnp.dot(matrix, matrix.T)
    return result

# JAX의 JIT 컴파일로 최적화
generate_and_multiply_jit = jax.jit(generate_and_multiply)

# 실행 시간 측정 함수
def measure_execution_time(func, num_runs=100):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        # 함수를 호출하여 실행
        func()
        jax.block_until_ready(func())  # 비동기 실행이 완료될 때까지 대기
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / num_runs
    return avg_time

# 평균 실행 시간 측정
avg_time = measure_execution_time(generate_and_multiply_jit, num_runs=40000)
print(f"Average execution time: {avg_time:.5f} seconds")
