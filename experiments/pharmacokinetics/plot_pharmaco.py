import numpy as np
import matplotlib.pyplot as plt
#from tueplots import bundles
#plt.rcParams.update(bundles.icml2022())

means_opt = []
means_eq= []
std_eq = []
std_opt = []
std_eq2 = []
std_opt2 = []
vals = range(2,22,2)
for i in vals:
	optimized = np.loadtxt("data/optimized"+str(i)+".txt")
	equally_spaced = np.loadtxt("data/equally_spaced"+str(i)+".txt")

	print ("Optimized:")
	print (np.median(optimized), np.std(optimized))
	means_opt.append(np.median(optimized))
	std_opt.append(np.quantile(optimized,0.25))
	std_opt2.append(np.quantile(optimized,0.75))

	print ("Eq.s.:")
	print (np.median(equally_spaced), np.std(equally_spaced))
	means_eq.append(np.median(equally_spaced))
	std_eq.append(np.quantile(equally_spaced,0.25))
	std_eq2.append(np.quantile(equally_spaced,0.75))

plt.semilogy(vals,means_opt,"-o", label = 'optimized design', color = "tab:blue")
plt.fill_between(vals, np.array(std_opt),np.array(std_opt2),alpha = 0.5,color = 'tab:blue')
plt.plot(vals,means_eq, "-o",label = 'equally spaced design', color = 'tab:orange')
plt.fill_between(vals, np.array(std_eq),np.array(std_eq2),alpha = 0.5,color = 'tab:orange')
plt.xlabel("data points")
plt.ylabel("$||\\hat{\\gamma}-\\gamma||_2^2$")
plt.legend()
plt.savefig("mle-pharmaco.png")
plt.show()