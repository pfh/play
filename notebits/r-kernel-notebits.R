
# Utilities when using Jupyter R kernel

cat("Loading utility functions, see https://github.com/pfh/play/blob/master/notebits/r-kernel-notebits.R\n")

library(IRdisplay)
library(ggplot2)

theme <- theme_bw()

show.plot <- function(filename, item, width=6.5,height=5) {
    png(sprintf("%s.png",filename), width=width,height=height,units="in",res=120)
    value <- item()
    dev.off()
    
    # ?<time> is a cache breaker
    display_html(
        sprintf("<p><img src=\"%s.png?%s\" width=%d height=%d/>",filename,Sys.time(),width*60,height*60)) 
    
    #postscript(sprintf("%s.eps",filename), width=width,height=height)
    #item()
    #dev.off()
    #display_html(sprintf("<p><a download href=\"%s.eps\">%s.eps</a>",filename,filename))
    
    # PDF is fine in Illustrator, and supports transparency

    pdf(sprintf("%s.pdf",filename), width=width,height=height)
    item()
    dev.off()
    display_html(sprintf(" <a download href=\"%s.pdf\">%s.pdf</a>",filename,filename))

    invisible(value)
}

show.ggplot <- function(filename, item, ...) {
    show.plot(filename, function() print(item), ...)
}

