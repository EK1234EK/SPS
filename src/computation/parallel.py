import multiprocessing

class Parallel_Pipeline:
    def __init__(self, sc_list, fun, cores):
        self.sc_input_list = sc_list
        self.sc_output_lst = []
        self.fun = fun
        self.pool = multiprocessing.Pool(cores)

    def run_pipeline(self):
        self.sc_output_lst = self.pool.map(self.pipeline_fun, range(len(self.sc_input_list)))
        return self.sc_output_lst

    def pipeline_fun(self, i):
        self.fun(self.sc_input_list[i])