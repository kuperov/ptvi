if (!require('stochvol'))
  install.packages('stochvol')
if (!require('rjson'))
  install.packages('rjson')


mcmc_SV <- function(experiment_file, outfile) {
  library(stochvol)
  library(rjson)
  
  experiment <- fromJSON(file = experiment_file)
  set.seed(experiment$algo_seed)

  fit <- svsample(experiment$y, designmatrix = 'ar1')
  summary(fit, showlatent = FALSE)
  #vp <- volplot(fit, forecast=10)

  # produce N forecasts
  N <- experiment$n
  t <- experiment$t
  draw_idxs <- sample(1:nrow(fit$para), size = N)
  mus <- fit$para[draw_idxs,'mu']
  phis <- fit$para[draw_idxs,'phi']
  sigmas <- fit$para[draw_idxs,'sigma']
  # project volatilities
  z_Ts <- fit$latent[draw_idxs,t-1]
  z_fcs <- rnorm(N, mean = mus + phis * (z_Ts - mus), sd = sigmas)
  # associated y projection
  y_fcs <- rnorm(N, 0, exp(z_fcs/2))
  fc_dens <- density(y_fcs)
  fc_pdf <- approxfun(fc_dens$x, fc_dens$y)  # interpolate kernel density pdf
  
  # we already have N samples from 1-step-ahead forecasts, sooo...
  ys <- experiment$y_next
  ds <- fc_pdf(ys)
  log_scores <- log(ds[ds > 0])
  log_score <- mean(log_scores, na.rm = TRUE)
  
  cat(sprintf('Log score = %.4f (sd = %.4f, N = %d)\n', 
                log_score, sd(log_scores), N))
  
  summ <- list(method = 'MCMC',
               summ = summary(fit),
               scores = log_scores,
               score = log_score,
               fc_draws = y_fcs,
               z_fcs = z_fcs,
               t = t,
               N = N)
  write(toJSON(summ), file = outfile)
  summ
}
