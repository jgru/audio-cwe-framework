# Information about the audio-cwe-framework

Implementation of a histogram-based watermarking method, which is commutative to a permutation cipher in the time domain. Furtheron a minimum knowledge verification in form of a probabillistic protocol, which is based on the graph isomorphism problem, is implemented. A detailed description can be found in the following journal article of Prof. Dr. R. Schmitz and myself: 
https://www.hindawi.com/journals/am/2017/5879257/abs/


The Python code follows the PEP8 styleguide and the recommendations of this site: https://gist.github.com/sloria/7001839

Structure
---
./audio-cwe-framework 
- core - contains the classes and modules, which implement the watermarking schemes and the asym verification protocol
- experimental_testing - contains code, that was used to generate the experimental results
- notebooks - contains IPython notebooks (which can be run in a browser e.g.), that were used in the course of
      exploration activities, demonstrate the usage of the classes and modules in core or show  certain other features
- plotting - contains code, which was used to plot the figures of the associated thesis

---

./res should contain soundfiles(marked and/or unmarked) 
- demo - results of marking processes
- testing - files for evaluation
