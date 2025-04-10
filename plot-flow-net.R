library(ggplot2)
library(scales)
library(dplyr)
library(tidyr)

thinning = seq(1, 10000, by = 10)

edges = c("01", "24", "36", "46", "56", "02", "13", "14", "25", "45")
names(edges) <- paste0("z", 1:10)
methods = c("Gibbs posterior", "Gibbs posterior without gradient",
    "Gradient-bridged posterior"
)

# ==================
# beta samples
# ==================

beta_truth = c(2.9433, 2.8440, 3.2228, 1.2193, 4.0797,
               6.1826, 4.1528, 2.1969, 2.5417, 2.7563)

beta_samples3 = read.table(
    "res/flow_net_beta_samples_grad-bridge.txt", sep = ","
)[thinning, ]
beta_samples1 = read.table(
    "res/flow_net_beta_samples_gibbs.txt", sep = ","
)[thinning, ]
beta_samples2 = read.table(
    "res/flow_net_beta_samples_kernel-h.txt", sep = ","
)[thinning, ]
beta_samples = rbind(beta_samples1, beta_samples2, beta_samples3)
colnames(beta_samples) <- paste0("z", 1:10)

df_beta <- as.data.frame(beta_samples) %>%
    mutate(method = factor(rep(methods, each = 1000), levels = methods))


# ==================
# density plots
# ==================

for (i in c(1:10)) {
    ggplot(df_beta, aes(x = method, y = !!sym(paste0("z", i)), fill = method)) +
        geom_violin(alpha = 0.8) +
        geom_hline(
            aes(yintercept = beta_truth[i]),
            linetype = "dashed", color = "black"
        ) +
        theme_bw() + theme(legend.position = "none") +
        labs(y = bquote(beta[.(edges[i])]), x = NULL) +
        theme(
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.title.y = element_text(size = 9),
            axis.text.y = element_text(size = 8)
        )

    ggsave(paste0("violin_flow_beta", i, ".png"),
        width=1.5, height=1, units="in"
    )
}





# ==================
# z samples
# ==================

z_truth = c(2.9433, 2.7573, 2.9433, 1.2193, 4.0797,
            5.299, 2.9433, 0.00, 2.5417, 1.538)

z_samples3 = read.table(
    "res/flow_net_z_samples_grad-bridge.txt", sep = ","
)[thinning, ]
z_samples1 = read.table(
    "res/flow_net_z_samples_gibbs.txt", sep = ","
)[thinning, ]
z_samples2 = read.table(
    "res/flow_net_z_samples_kernel-h.txt", sep = ","
)[thinning, ]
z_samples = rbind(z_samples1, z_samples2, z_samples3)
z_samples[z_samples < 0] = 0
colnames(z_samples) <- paste0("z", 1:10)

df_z <- as.data.frame(z_samples) %>%
    mutate(method = factor(rep(methods, each = 1000), levels = methods))

for (i in c(1:10)) {
    ggplot(df_z, aes(x = method, y = !!sym(paste0("z", i)), fill = method)) +
        geom_violin(alpha = 0.8) +
        geom_hline(
            aes(yintercept = z_truth[i]),
            linetype = "dashed", color = "black"
        ) +
        theme_bw() + theme(legend.position = "none") +
        labs(y = bquote(z[.(edges[i])]), x = NULL) +
        # scale_y_continuous(limits = c(0, 0.04)) +
        theme(
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.title.y = element_text(size = 9),
            axis.text.y = element_text(size = 8)
        )

    ggsave(paste0("violin_flow_z", i, ".png"),
        width=1.5, height=1, units="in"
    )
}



# =========================
# # Plot histograms for each dimension
# =========================


# z_lim = matrix(c(apply(z_samples, 2, min), apply(beta_samples, 2, max)), ncol=2)
# z_lim[z_lim < 0] = 0
# z_lim[which(z_lim > 8 & (row(z_lim) != 6))] = 8
# z_lim[6, 2] = 11
# beta_lim = z_lim
# z_lim[8, 2] = 0.02

# Plot histograms for each dimension

# nbreaks = rep(5, 10)
# nbreaks[4] = 4
# nbreaks[6] = 4

# for(i in c(1:10)) {
#     ggplot(
#         data = data.frame(x = z_samples[thinning, i]),
#         aes(x = x, y = after_stat(density))) +
#     geom_histogram(bins=50) + theme_bw() + 
#     xlab(bquote(z[.(edges[i])])) + ylab('Density') +
#     geom_vline(xintercept = z_truth[i], color = '#ef609f', linewidth = 1.2) +
#     theme(
#         axis.title.x = element_text(size = 15),
#         axis.title.y = element_text(size = 14),
#         axis.text.x = element_text(size = 12),
#         axis.text.y = element_text(size = 12)
#     ) +
#     theme(plot.margin = margin(8, 15, 3, 3)) +
#     scale_x_continuous(
#         limits = c(z_lim[i, 1], z_lim[i, 2]), n.breaks = nbreaks[i])

#     ggsave(paste("hist_flow_z", i, ".png", sep = ""),
#             width = 2.2, height = 1.5, units = "in")
# }


# nbreaks = rep(5, 10)
# nbreaks[4] = 4

# for (i in c(1:10)) {
#     ggplot(
#         data = data.frame(x = beta_samples[thinning, i]),
#         aes(x = x, y = after_stat(density))) +
#     geom_histogram(bins=50) + theme_bw() +
#     xlab(bquote(beta[.(edges[i])])) + ylab('Density') +
#     geom_vline(xintercept = beta_truth[i], color = '#ef609f', linewidth = 1.2) +
#     theme(
#         axis.title.x = element_text(size = 15),
#         axis.title.y = element_text(size = 14),
#         axis.text.x = element_text(size = 12),
#         axis.text.y = element_text(size = 12)
#     ) +
#     theme(plot.margin = margin(8, 15, 3, 3)) +
#     scale_x_continuous(
#         limits = c(beta_lim[i, 1], beta_lim[i, 2]), n.breaks = nbreaks[i])

#     ggsave(
#         paste("hist_flow_beta", i, ".png", sep=""),
#         width=2.2, height=1.5, units="in")
# }