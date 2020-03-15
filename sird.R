# Global setup
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Load data from the web
## confirmed <- read.csv(url('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'), stringsAsFactors = FALSE)
## dead <- read.csv(url('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'), stringsAsFactors = FALSE)
## recovered <- read.csv(url('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'), stringsAsFactors = FALSE)
confirmed <- read.csv('time_series_19-covid-Confirmed.csv')
dead <- read.csv('time_series_19-covid-Deaths.csv')
recovered <- read.csv('time_series_19-covid-Recovered.csv')

data_start_index = 5

count_timesteps <- function(set) {
    return(length(set[1,]) - data_start_index + 1)
}

extract_measurements <- function(set, name) {
    idx = set$Country.Region == name
    return(set[idx, data_start_index:length(set[idx,])])
}

# Italy population: 60488135
country_name = "Italy"
country_population = 60488135
i0 = extract_measurements(confirmed, country_name)
r0 = extract_measurements(recovered, country_name)
d0 = extract_measurements(dead, country_name)
s0 = country_population - i0 - r0 -d0

y = rbind(s0, i0, r0, d0)
T <- count_timesteps(confirmed)
ts <- 0:(count_timesteps(confirmed) - 1)

fit_data = list(y=t(y), T=T, ts=ts)

fit <- stan(file='sird_reduced.stan', data=fit_data,
            seed=2222, chains=8, iter=3000,
            control=list(adapt_delta=0.9))

check_hmc_diagnostics(fit)
print(fit)