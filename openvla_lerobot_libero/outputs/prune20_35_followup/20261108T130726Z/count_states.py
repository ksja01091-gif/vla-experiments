from libero.libero import benchmark

suite = benchmark.get_benchmark_dict()['libero_spatial']()
print('TASK0_INITIAL_STATES', len(suite.get_task_init_states(0)))
