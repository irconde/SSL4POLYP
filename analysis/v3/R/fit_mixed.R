# SAP v3.3 section 4 (variant "ttest", passed by analysis/v3): one-sample t-test on the K partition means.
# SAP v3.2c section 4 (variant "satterthwaite"): d_{k,s} = mu + u_k + e_{k,s}, REML (lme4), Satterthwaite df.
# Fallback (section 4.3): if the fit is singular (partition variance estimated as 0) or does not
# converge, average the seeds within each partition and run a one-sample t-test on the K means (df = K - 1).
#
# Usage: Rscript fit_mixed.R <input.csv: id,k,s,d> <output.csv> [variant]
#   variant "satterthwaite" (default when no variant is given; SAP v3.2c): lmer with Satterthwaite df, fallback as above
#           "kenward-roger": the same with Kenward-Roger df and adjusted SE (pbkrtest), fallback as above
#           "ttest": always the one-sample t-test on the K partition means (df = K - 1)
# One fit per id. With a single partition (K = 1) the model is not estimable and a one-sample t-test
# over the seeds is reported instead (method "t_seeds_K1", used for the k = 1 sensitivity analysis).

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

SINGULAR_TOL <- 1e-4  # lme4::isSingular default

args <- commandArgs(trailingOnly = TRUE)
VARIANT <- if (length(args) >= 3) args[3] else "satterthwaite"
stopifnot(VARIANT %in% c("satterthwaite", "kenward-roger", "ttest"))
DDF <- if (VARIANT == "kenward-roger") "Kenward-Roger" else "Satterthwaite"
input <- read.csv(args[1], colClasses = c(id = "character", k = "character", s = "character", d = "numeric"))

empty_row <- function(id, n, K, est) {
  data.frame(id = id, n = n, K = K, method = NA_character_, estimate = est, se = NA_real_, df = NA_real_,
             t = NA_real_, p = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_, var_partition = NA_real_,
             var_seed = NA_real_, singular = NA, converged = NA, note = "", stringsAsFactors = FALSE)
}

ttest_into <- function(res, values, method) {
  if (length(values) < 2) {
    res$method <- method
    res$note <- paste(res$note, "fewer than 2 values; no test")
    return(res)
  }
  if (sd(values) == 0) {
    res$method <- "constant"
    res$estimate <- mean(values)
    res$se <- 0
    res$note <- paste(res$note, "zero variance; no test")
    return(res)
  }
  tt <- t.test(values, mu = 0)
  res$method <- method
  res$estimate <- unname(tt$estimate)
  res$se <- unname(tt$stderr)
  res$df <- unname(tt$parameter)
  res$t <- unname(tt$statistic)
  res$p <- tt$p.value
  res$ci_lo <- tt$conf.int[1]
  res$ci_hi <- tt$conf.int[2]
  res
}

fit_one <- function(x) {
  x <- x[is.finite(x$d), ]
  x$k <- factor(x$k)
  n <- nrow(x)
  K <- nlevels(x$k)
  res <- empty_row(x$id[1], n, K, if (n > 0) mean(x$d) else NA_real_)
  if (n == 0) {
    res$method <- "no_data"
    return(res)
  }
  if (K < 2) {
    return(ttest_into(res, x$d, "t_seeds_K1"))
  }
  partition_means <- tapply(x$d, x$k, mean)
  if (sd(x$d) == 0) {
    return(ttest_into(res, as.numeric(partition_means), "fallback_t"))
  }
  if (VARIANT == "ttest") {
    return(ttest_into(res, as.numeric(partition_means), "partition_means_t"))
  }
  warns <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      lmerTest::lmer(d ~ 1 + (1 | k), data = x, REML = TRUE,
                     control = lmerControl(check.conv.singular = "ignore")),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      warns <<- c(warns, paste("error:", conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) {
    res$converged <- FALSE
    res$note <- paste(warns, collapse = "; ")
    return(ttest_into(res, as.numeric(partition_means), "fallback_t"))
  }
  conv_msgs <- fit@optinfo$conv$lme4$messages
  converged <- is.null(conv_msgs) && fit@optinfo$conv$opt == 0 && !any(grepl("converge", warns, ignore.case = TRUE))
  singular <- isSingular(fit, tol = SINGULAR_TOL)
  vc <- as.data.frame(VarCorr(fit))
  res$var_partition <- vc$vcov[vc$grp == "k"]
  res$var_seed <- vc$vcov[vc$grp == "Residual"]
  res$singular <- singular
  res$converged <- converged
  res$note <- paste(c(warns, conv_msgs), collapse = "; ")
  if (singular || !converged) {
    return(ttest_into(res, as.numeric(partition_means), "fallback_t"))
  }
  co <- summary(fit, ddf = DDF)$coefficients
  res$method <- if (VARIANT == "kenward-roger") "lmer_kr" else "lmer"
  res$estimate <- co["(Intercept)", "Estimate"]
  res$se <- co["(Intercept)", "Std. Error"]
  res$df <- co["(Intercept)", "df"]
  res$t <- co["(Intercept)", "t value"]
  res$p <- co["(Intercept)", "Pr(>|t|)"]
  q <- qt(0.975, res$df)
  res$ci_lo <- res$estimate - q * res$se
  res$ci_hi <- res$estimate + q * res$se
  res
}

ids <- unique(input$id)
out <- do.call(rbind, lapply(ids, function(i) fit_one(input[input$id == i, ])))
write.csv(out, args[2], row.names = FALSE)
