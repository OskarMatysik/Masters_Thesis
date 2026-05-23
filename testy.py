from src.models import DeffuantWeisbuchModel

if __name__ == "__main__":
    test1 = DeffuantWeisbuchModel(N=1000, d=0.5, mu=0.15, t=100, topology="full")
    test2 = DeffuantWeisbuchModel(N=1000, d=0.3, mu=0.18, t=100, topology="full")
    test3 = DeffuantWeisbuchModel(N=1000, d=0.2, mu=0.2, t=100, topology="full")
    test4 = DeffuantWeisbuchModel(N=1000, d=0.1, mu=0.3, t=100, topology="full")
    test5 = DeffuantWeisbuchModel(N=1000, d=0.05, mu=0.5, t=100, topology="full")
    test1.run()
    test2.run()
    test3.run()
    test4.run()
    test5.run()
    test1.plot_time_chart()
    test2.plot_time_chart()
    test3.plot_time_chart()
    test4.plot_time_chart()
    test5.plot_time_chart()