# Accelerometer Based Eveness Estimation

This repository contains the code to replicate the findings of ....
Including two datasets: ZEB Data Set and Windshield Data Set.

## Code
The code features Accelerometer Signal Processing Pipelines (ASPP), which are a means to analyze accelerometer signals.
The code automatically evaluates different ASPP configurations in order to optimize the setup score and the unevenness score.

- The setup score evaluates the consitency of the aspp inbetween different sensor setups, where we expect the ASPP to produce consitent results regadless of senosr setup.
- The unevenness score evaluates the correlation of the ASPP output with the underlying ZEB grade (which is a measure for longitudinal road evenness)

## ZEB Data Set
Contains accelerometer signals with a corresponding ZEB Grade attached.

## Windshield Data Set
Contains accelerometer signals recorded on the same road, with different parameters in the windshiled setup changed.
The data set provides insights into the effects of different windshield mounting parameters.