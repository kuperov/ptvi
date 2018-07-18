library(tidyverse)
library(stochvol)

fx <- read_csv('sparse_vi/data/DEXUSUK.csv',
               col_types = cols(
                  DATE = col_date(),
                  DEXUSUK = col_number()
                ),
                na = '.') %>%
  rename(USD_per_GBP = DEXUSUK) %>%
  filter(!is.na(USD_per_GBP))

sv <- svsample(fx$USD_per_GBP, designmatrix = 'ar1', )
summary(sv, showlatent = FALSE)

volplot(sv, dates = fx$DATE[-1], forecast=100)
write_csv(as.data.frame(sv$latent), 'sparse_vi/data/gbp_usd_latent_draws.csv')
write_csv(as.data.frame(sv$para), 'sparse_vi/data/gbp_usd_parameter_draws.csv')
