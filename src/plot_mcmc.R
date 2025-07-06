library(ggplot2)

# the function for plotting the trace of the MCMC samples
plot_trace <- function(
    res, yname, filename, width=1.7, height=1) {
    num_samples <- length(res)
    df <- data.frame(b = c(res), Iteration = c(1:num_samples))

    ggplot(data = df, aes(x = Iteration, y = b)) +
        geom_line() + theme_bw() +
        theme(legend.position = "none") + ylab(yname) +
        theme(
            plot.margin = margin(t = 4, r = 12, b = 3, l = 3),
            axis.title.x = element_text(size = 11),
            axis.title.y = element_text(size = 11),
            axis.text.x = element_text(size = 8),
            axis.text.y = element_text(size = 8)
        ) +
        scale_x_continuous(breaks = c(0, 3000, 6000, 10000))

    ggsave(filename, width = width, height = height, units = 'in')
}