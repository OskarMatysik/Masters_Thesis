from src.models import *

if __name__ == "__main__":
    # model_1 = DeffuantWeisbuchModel(N=2000, d=0.5, mu=0.5, t=50)
    model_2 = DeffuantWeisbuchModel(N=2000, d=0.2, mu=0.5, t=100, topology="scale-free")
    # model_1.run()
    model_2.run()
    # model_1.plot_time_chart()
    model_2.plot_time_chart()
    # model_1.plot_final_vs_initial()
    model_2.plot_final_vs_initial()
    model_2.statistics()
    # breakpoint()