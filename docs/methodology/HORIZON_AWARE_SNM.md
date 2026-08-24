# Horizon-Aware SNM (HA-SNM)

## Motivation

Standard Signed Neuron with Memory (SNM) emits a negative spike when the
membrane deficit crosses `-theta` and the neuron has positive transmitted
credit.  This corrects asynchronous over-firing, but applies the same decision
boundary at every time step and every inference horizon.  In the repository's
four-model ablations, that fixed rule gives useful low-step corrections but can
be neutral or mildly harmful at T=16/32.

HA-SNM changes only the negative-spike decision boundary.  It keeps the
original membrane, transmitted-credit state, R0 rule, positive threshold and
negative event amplitude.

## Rule

Let `p=t/(T-1)` and

```text
g(T) = min(1, T_ref / T)
lambda(t,T) = 1 + g(T) * [(lambda_start - 1)(1-p) + (lambda_end - 1)p]
```

The frozen deployment constants are:

```text
lambda_start = 1.25
lambda_end = 0.50
T_ref = 8
```

For T>1, HA-SNM emits

```text
-theta  if membrane <= -lambda(t,T)*theta and transmitted > 0
0       otherwise
```

At T=1 the credit condition prevents negative spikes, exactly as in standard
SNM.

The schedule has two complementary effects:

1. Early time steps require stronger evidence before undoing a positive spike,
   reducing premature positive-negative churn.
2. Near a short inference horizon the boundary is relaxed, allowing residual
   over-firing to be corrected after more temporal evidence has arrived.
3. The whole schedule contracts toward standard SNM by `8/T` at long horizons,
   so the modification does not remain aggressively tuned for low latency.

## Complexity

- Additional dense per-neuron state: 0 bytes.
- Reused state: existing `mem` and `transmitted` tensors.
- Logical deployment constants: three scalars, 12 bytes if stored as FP32.
  The PyTorch reference keeps non-tensor copies on each `SignedIF` object; they
  are absent from `state_dict` and do not create activation-sized storage.
- Negative event amplitude: unchanged at `-theta`.
- Event overhead: measured directly using negative-spike rates and input-driven
  SOPs in the formal report.

The scalar threshold can be precomputed once for every `(layer,T,t)` and does
not depend on batch size or spatial resolution.

## Parameter selection protocol

Nine small schedules were compared on the frozen 1,000-image calibration
validation data for CIFAR-10/100 and ResNet20/VGG16.  Test images were not
evaluated by the screening script.  The single global schedule above was
frozen before the four formal 10,000-image test runs.  Test results never alter
the schedule.

## Scope of the claim

"HA-SNM" is a repository method name and an experimentally evaluated design.
It is not a claim of patent novelty or an exhaustive literature search.  The
baseline SNM mechanism follows Wang et al., *Signed Neuron with Memory: Towards
Simple, Accurate and High-Efficient ANN-SNN Conversion*, IJCAI 2022:
https://www.ijcai.org/proceedings/2022/347.
