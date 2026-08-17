accurate as of 8/14/26
ic_gbm.csv contains the out of sample information coefficient (the spearman rank correlation) for the realized returns ranking vs. prediction rankings.
Because each iteration of a walk-forward produces one cross-sectional vector of predictions, this ic_gmb.csv should have as many rows as there are iterations of walk-forwards.

8/14/26
I tried to figure out what metric to use would be best
I tried to see how well the model does in terms of "for the stocks the model predicted would be in the top decile, how did those stocks do?" I compared with both NCDG and pairwise but neither of them were great
I'm wondering if there is a fundamental investment thesis here and I realized that I didn't even base this on an investment thesis
figured out step_months=HORIZON has to be true, since we want to claim that our OOS IC performance are independent observations
I'm thinking I need to pick better features


8/16/26
Implemented a couple new features that might be less colinear
Realized that n_estimators being a bit smaller makes a big difference. at first, 7 seemed optimal, but read more below
Also tried lower learning rate to 0.001
Implemented tail-only IC
Then, n_estimator = 12 was optimal for tail_ic_mean