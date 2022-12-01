#pragma once

#define LAMBDA_ADD [](scalar_t a, scalar_t b) { return a + b; }
#define LAMBDA_MUL [](scalar_t a, scalar_t b) { return a * b; }
#define LAMBDA_DIV [](scalar_t a, scalar_t b) { return a / b; }
#define LAMBDA_POW [](scalar_t a, scalar_t b) { return pow(a, b); }
#define LAMBDA_MAX [](scalar_t a, scalar_t b) { return std::max(a, b); }
#define LAMBDA_EQ [](scalar_t a, scalar_t b) { return a == b; }
#define LAMBDA_GE [](scalar_t a, scalar_t b) { return a >= b; }
#define LAMBDA_LOG [](scalar_t a) { return log(a); }
#define LAMBDA_EXP [](scalar_t a) { return exp(a); }
#define LAMBDA_TANH [](scalar_t a) { return tanh(a); }