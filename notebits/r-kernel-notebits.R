
# Utilities when using Jupyter R kernel

cat("Loading utility functions, see https://github.com/pfh/play/tree/master/notebits/r-kernel-notebits.R\n")

show.plot <- function(filename, item, width=6.5,height=5) {
    png(sprintf("%s.png",filename), width=width,height=height,units="in",res=120)
    value <- item()
    dev.off()
    pyexec(sprintf("display(Image('%s.png', retina=True))", filename)) 
    
    postscript(sprintf("%s.eps",filename), width=width,height=height)
    item()
    dev.off()
    pyexec(sprintf('display(HTML("<p><a download href=\\"%s.eps\\">%s.eps</a>"))', filename,filename))

    invisible(value)
}

show.ggplot <- function(filename, item, ...) {
    show.plot(filename, function() print(item), ...)
}

