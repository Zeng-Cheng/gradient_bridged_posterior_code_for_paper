# ==============================================
# plot the trace and acf for mcmc with chosen mass
# ==============================================


library(ggplot2)
library(scales)

beta_samples = read.table(
    "res/flow_net_beta_samples_grad-bridge.txt", sep = ","
)
z_samples = read.table(
    "res/flow_net_z_samples_grad-bridge.txt", sep = ","
)

edges = c("01", "24", "36", "46", "56", "02", "13", "14", "25", "45")

# ==================
# trace plots
# ==================

for (i in c(1:10)) {
    trace = data.frame(x = c(z_samples[, i]), Iteration = c(1:10000))

    ggplot(data = trace, aes(x = Iteration, y = x)) +
        geom_line(linewidth=0.1) + theme_bw() + ylab(bquote(z[.(edges[i])])) +
        theme(plot.margin = margin(4, 12, 3, 3)) +
        theme(
            axis.title.x = element_text(size = 11),
            axis.title.y = element_text(size = 11),
            axis.text.x = element_text(size = 8),
            axis.text.y = element_text(size = 8)) +
        scale_x_continuous(breaks = c(0, 3000, 6000, 10000))

    ggsave(paste("trace_flow_z", i, ".png", sep=""),
        width=1.7, height=1, units='in')
}

for (i in c(1:10)) {
    trace = data.frame(x = c(beta_samples[, i]), Iteration = c(1:10000))

    ggplot(data = trace, aes(x = Iteration, y = x)) +
        geom_line(linewidth=0.1) + theme_bw() + ylab(bquote(beta[.(edges[i])])) +
        theme(plot.margin = margin(4, 12, 3, 3)) +
        theme(
            axis.title.x = element_text(size = 11),
            axis.title.y = element_text(size = 11),
            axis.text.x = element_text(size = 8),
            axis.text.y = element_text(size = 8)) +
        scale_x_continuous(breaks = c(0, 3000, 6000, 10000))

    ggsave(paste("trace_flow_beta", i, ".png", sep=""),
             width=1.6, height=1, units='in')
}


# ==================
# acf plots
# ==================

# lag_max = 40
# for (i in c(1:10)) {

#     acf_df <- data.frame(
#         ACF = c(
#             as.numeric(acf(z_samples[thinning, i],
#                 lag.max=lag_max, plot = FALSE)[[1]])
#         ),
#         Lag = c(0:lag_max)
#     )

#     ggplot(data = acf_df, aes(x = Lag, y = ACF)) +
#     geom_bar(stat = "identity", width = 0.5) + theme_bw() +
#     theme(
#         axis.title.x = element_text(size = 24),
#         axis.title.y = element_text(size = 24),
#         axis.text.x = element_text(size = 16),
#         axis.text.y = element_text(size = 16))

#     ggsave(paste("acf_flow_z", i, ".png", sep = ""),
#         width = 3.3, height = 2, units = 'in')
# }


# plot box plots of ACFs of all z
p <- ncol(z_samples)
num_iters <- nrow(z_samples)
lag_max <- 40
thinned_idx <- seq(1, num_iters, by = 10)
acf_w <- c()
for (i in 1:p) {
    cur_acf <- as.numeric(acf(
        z_samples[thinned_idx, i],
        lag.max = lag_max, plot = FALSE)[[1]])
    acf_w <- c(acf_w, cur_acf)
}

df_acf_w <- data.frame(
    ACF = acf_w,
    Lag = c(0:lag_max),
    dim = factor(rep(1:p, each = lag_max + 1))
)

ggplot(data = df_acf_w, aes(Lag, ACF, group = Lag)) +
geom_boxplot() + theme_bw() +
theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))

ggsave("acf_flow_z.png", width=5, height=2.2, units='in')



# lag_max = 40
# for (i in c(1:10)) {

#     acf_df <- data.frame(
#         ACF = c(
#             as.numeric(acf(beta_samples[thinning, i],
#                 lag.max=lag_max, plot = FALSE)[[1]])
#         ),
#         Lag = c(0:lag_max)
#     )

#     ggplot(data = acf_df, aes(x = Lag, y = ACF)) +
#     geom_bar(stat = "identity", width = 0.5) + theme_bw() +
#     theme(
#         axis.title.x = element_text(size = 24),
#         axis.title.y = element_text(size = 24),
#         axis.text.x = element_text(size = 16),
#         axis.text.y = element_text(size = 16))

#     ggsave(paste("acf_flow_beta", i, ".png", sep=""),
#         width=3.3, height=2, units='in')
# }

# plot box plots of ACFs of all beta
p <- ncol(beta_samples)
num_iters <- nrow(beta_samples)
lag_max <- 40
thinned_idx <- seq(1, num_iters, by = 10)
acf_w <- c()
for (i in 1:p) {
    cur_acf <- as.numeric(acf(
        beta_samples[thinned_idx, i],
        lag.max = lag_max, plot = FALSE)[[1]])
    acf_w <- c(acf_w, cur_acf)
}

df_acf_w <- data.frame(
    ACF = acf_w,
    Lag = c(0:lag_max),
    dim = factor(rep(1:p, each = lag_max + 1))
)

ggplot(data = df_acf_w, aes(Lag, ACF, group = Lag)) +
geom_boxplot() + theme_bw() +
theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))

ggsave("acf_flow_beta.png", width=5, height=2.2, units='in')




# ==============================================
# plot the trace and acf for mcmc with no mass
# ==============================================





library(ggplot2)
library(scales)

beta_samples = read.table(
    "res/flow_net_beta_samples_grad-bridge-nomass.txt", sep = ","
)
z_samples = read.table(
    "res/flow_net_z_samples_grad-bridge-nomass.txt", sep = ","
)

edges = c("01", "24", "36", "46", "56", "02", "13", "14", "25", "45")
print(dim(z_samples))

# ==================
# trace plots
# ==================

for (i in c(1:10)) {
    trace = data.frame(x = c(z_samples[, i]), Iteration = c(1:10000))

    ggplot(data = trace, aes(x = Iteration, y = x)) +
        geom_line(linewidth=0.1) + theme_bw() + ylab(bquote(z[.(edges[i])])) +
        theme(plot.margin = margin(4, 12, 3, 3)) +
        theme(
            axis.title.x = element_text(size = 11),
            axis.title.y = element_text(size = 11),
            axis.text.x = element_text(size = 8),
            axis.text.y = element_text(size = 8)) +
        scale_x_continuous(breaks = c(0, 3000, 6000, 10000))

    ggsave(paste("trace_flow_z_nomass", i, ".png", sep=""),
        width=1.7, height=1, units='in'
    )
}


for (i in c(1:10)) {
    trace = data.frame(x = c(beta_samples[, i]), Iteration = c(1:10000))

    ggplot(data = trace, aes(x = Iteration, y = x)) +
        geom_line(linewidth=0.1) + theme_bw() + ylab(bquote(beta[.(edges[i])])) +
        theme(plot.margin = margin(4, 12, 3, 3)) +
        theme(
            axis.title.x = element_text(size = 11),
            axis.title.y = element_text(size = 11),
            axis.text.x = element_text(size = 8),
            axis.text.y = element_text(size = 8)) +
        scale_x_continuous(breaks = c(0, 3000, 6000, 10000))

    ggsave(paste("trace_flow_beta_nomass", i, ".png", sep=""),
             width=1.6, height=1, units='in')
}


# ==================
# acf plots
# ==================


# plot box plots of ACFs of all z

p <- ncol(z_samples)
num_iters <- nrow(z_samples)
lag_max <- 40
thinned_idx <- seq(1, num_iters, by = 10)
acf_w <- c()
for (i in 1:p) {
    cur_acf <- as.numeric(acf(
        z_samples[thinned_idx, i],
        lag.max = lag_max, plot = FALSE
    )[[1]])
    acf_w <- c(acf_w, cur_acf)
}

df_acf_w <- data.frame(
    ACF = acf_w,
    Lag = c(0:lag_max),
    dim = factor(rep(1:p, each = lag_max + 1))
)

ggplot(data = df_acf_w, aes(Lag, ACF, group = Lag)) +
    geom_boxplot() + theme_bw() +
    theme(
            axis.title.x = element_text(size = 15),
            axis.title.y = element_text(size = 15),
            axis.text.x = element_text(size = 14),
            axis.text.y = element_text(size = 14))

ggsave("acf_flow_z_nomass.png", width=5, height=2.2, units='in')



# plot box plots of ACFs of all beta
p <- ncol(beta_samples)
num_iters <- nrow(beta_samples)
lag_max <- 40
thinned_idx <- seq(1, num_iters, by = 10)
acf_w <- c()
for (i in 1:p) {
    cur_acf <- as.numeric(acf(
        beta_samples[thinned_idx, i],
        lag.max = lag_max, plot = FALSE)[[1]])
    acf_w <- c(acf_w, cur_acf)
}

df_acf_w <- data.frame(
    ACF = acf_w,
    Lag = c(0:lag_max),
    dim = factor(rep(1:p, each = lag_max + 1))
)

ggplot(data = df_acf_w, aes(Lag, ACF, group = Lag)) +
    geom_boxplot() + theme_bw() +
    theme(
            axis.title.x = element_text(size = 15),
            axis.title.y = element_text(size = 15),
            axis.text.x = element_text(size = 14),
            axis.text.y = element_text(size = 14))

ggsave("acf_flow_beta_nomass.png", width=5, height=2.2, units='in')