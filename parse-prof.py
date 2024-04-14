import cProfile
import pstats

stream = open('sac.txt', 'w')
stats = pstats.Stats('sac.prof', stream=stream)
stats.sort_stats('cumtime')
stats.print_stats()