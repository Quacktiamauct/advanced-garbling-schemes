# Advanced Garbling Schemes

## Current goals
- Use https://homes.esat.kuleuven.be/~nsmart/MPC/
- Rewrite YAO to use the above.
- Do the other two.
- ???
- Profit

## Project Description
### Group
- Frederik Munk Madsen
- Mikkel Wienberg Madsen

### Motivation
Garbling schemes are a way to securely evaluate boolean circuits between two parties. In the basic garbling scheme we have seen when a circuit is garbled we generate a truth table for each of the gates where the values are replaced by k-bit random strings (for some security parameter key). Garbling and evaluating a circuit requires computing some "encoding" function f to both garble and evaluate the circuit, additionally the garbled circuit must be somehow sent from the garbler to the evaluator. We would like to look at ways to optimize this process, by reducing both the size of the required truth tables and thus also the amount of data transferred between parties.

### Goal
Implementation of the two garbling schemes presented in ZRE15 and GLNP15 used for the blood type donation compatibility function from the handins and benchmarks and comparisons of the two schemes as well as the simple Yao implementation from Handin 5.

### Objectives
- Read and udnerstand the garbling schemes described in ZRE15 and GLNP15
- Implement the aforementioned garbling schemes and use them to compute blood type compatibility
- Benchmark and compare the implementation of the aforementioned garbling scehmes with eachother and the implementation from handin


## Papers
- [Fast Garbling of Circuits Under Standard Assumptions](https://eprint.iacr.org/2015/751.pdf)
- [Two Halves Make a Whole](https://eprint.iacr.org/2014/756.pdf)
- [Three Halves Make a Whole?](https://eprint.iacr.org/2021/749.pdf)



## Original Description
### Advanced Garbling Schemes:
In the course we have seen only very basic garbling schemes. More advanced schemes exist like [ZRE15, GLNP15] and a very recent one [RR21]. They are more advanced
because they allow to garble gates using less computation, because they produce shorter garbled gates,
or a combination of both. The project is about reading, understanding, implementing and benchmarking the first two garbling schemes. The third one (and potentially more) could be included depending
on your ambition and/or the size of the group.

- [ZRE15] Samee Zahur, Mike Rosulek, and David Evans. Two halves make a whole - reducing data transfer
          in garbled circuits using half gates. pages 220–250, 2015. 2
- [GLNP15] Shay Gueron, Yehuda Lindell, Ariel Nof, and Benny Pinkas. Fast garbling of circuits under
           standard assumptions. pages 567–578, 2015. 2
- [RR21] Mike Rosulek and Lawrence Roy. Three halves make a whole? Beating the half-gates lower
         bound for garbled circuits. pages 94–124, 2021. 2
