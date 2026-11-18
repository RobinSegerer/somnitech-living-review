# Optional independent cross-check against R/metafor.
# Run from project root: Rscript validation/metafor_crosscheck.R
# Packages: install.packages("metafor")

library(metafor)
d <- read.csv("data/studies.csv", fileEncoding = "UTF-8")
for (version in 0:4) {
  subset <- d[d$phase <= min(version, 3), ]
  if (version == 4) {
    i <- subset$study_id == "S06"
    subset$mean_int[i] <- subset$revised_mean_int[i]
    subset$sd_int[i] <- subset$revised_sd_int[i]
  }
  e <- escalc(measure = "MD", m1i = mean_int, sd1i = sd_int, n1i = n_int,
              m2i = mean_ctrl, sd2i = sd_ctrl, n2i = n_ctrl, data = subset)
  fit <- rma(yi, vi, data = e, method = "REML", test = "adhoc")
  cat(sprintf("V%d: k=%d MD=%+.3f CI=[%+.3f, %+.3f] tau2=%.3f\n",
              version, fit$k, as.numeric(fit$b), fit$ci.lb, fit$ci.ub, fit$tau2))
}
