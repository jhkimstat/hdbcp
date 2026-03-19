#ifndef INTERVALS_H
#define INTERVALS_H

#include <RcppArmadillo.h>

arma::umat generate_seeded_intervals(int n, double rho, double m_min);

#endif
