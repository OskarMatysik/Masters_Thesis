from src.models import *
from src.calibration import GA1Calibration

if __name__ == "__main__":
    # model_2 = DeffuantWeisbuchModel(N=2000, d=0.2, mu=0.5, t=100, topology="scale-free")
    # model_2.run()
    # model_2.plot_time_chart()
    # model_2.plot_final_vs_initial()
    # model_2.statistics()
    calibration = GA1Calibration(o_name="o_N1000_d0.2_mu0.5", s_name="s_N1000_d0.2_mu0.5", p_c=0.8, p_m=0.1, max_iter=200, stop_fitness=0.01, L_p=0.1, U_p=0.9)
    breakpoint()

