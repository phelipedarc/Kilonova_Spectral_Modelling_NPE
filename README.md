# Kilonova Spectral Modelling Using Amortized Neural Posterior Estimation: A Case Study with AT 2017gfo
Kilonovae are a class of astronomical transients observed as counterparts to binary neutron star and neutron star/black holes mergers. They serve as probes for heavy-element nucleosynthesis in astrophysical environments. These studies rely on inference of the physical parameters (e.g., ejecta mass, velocity, composition) that describe kilonovae based on electromagnetic observations, which is a complex inverse problem, typically tackled by sampling-based inference methods such as Markov-chain Monte Carlo (MCMC) or nested sampling technique. 
However,  repeated inferences can be computationally expensive due to the sequential nature of these methods. This poses a significant challenge to ensuring the reliability and statistical validity of the posterior approximations and thus the kilonova parameters themselves. We present a novel approach: Simulation-Based Inference (SBI) using simulations produced by \texttt{KilonovaNet}. Our method employs an ensemble of Amortized Neural Posterior Estimation (ANPE) with an embedding network to directly predict posterior distributions from simulated spectral energy distributions (SEDs). We take advantage of the quasi-instantaneous inference time of ANPE to demonstrate the reliability of our posterior approximations using diagnostics tools, including coverage diagnostic and posterior predictive checks.  We further test our model with real observations from AT\,2017gfo, the only kilonova with multi-messenger data, demonstrating agreement with previous likelihood-based methods while reducing inference time down to a few seconds. The inference results produced by ANPE appear to be conservative and reliable, paving the way for faster, testable and more efficient kilonova parameter inference.
