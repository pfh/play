
library(fitnoise) #Python interop
library(ggplot2)
library(vcd)

theme <- theme_bw()

heading <- function(...) {
    pyset_scalar("what", paste(..., sep="",collapse=""))
    pyexec("display(HTML('<h2>'+what+'</h2>'))")
}

say <- function(...) {
    pyset_scalar("what", paste(..., sep="",collapse=""))
    pyexec("print what")
}

say.print <- function(item) {
    pyset_scalar("what", paste(capture.output(item),collapse="\n"))
    pyexec("print what")
}

give.table <- function(filename, frame, comment="") {
    write.csv(frame, sprintf("%s.csv",filename))
    pyset_scalar("what", sprintf("<p><a href=\"%s.csv\">%s.csv</a> %s", filename, filename, comment))
    pyexec("display(HTML(what))")
}

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




report.cor <- function(x,y, method='pearson') {
    good <- !(is.na(x) | is.infinite(x) | is.na(y) | is.infinite(y))
    result <- cor.test(x[good],y[good],method=method)
    if (result$p.value < 0.0001) {
        sig <- 'p much < 0.001'
    } else {
        sig <- sprintf('p=%f', result$p.value)
    }
    sprintf("%.4f (%s, n=%d)", result$estimate, sig, sum(good))
}

show.scatter <- function(name, x,y, xlab, ylab, logx=F, logy=F, loggood=NULL) {
    heading(xlab, " vs ", ylab)

    xx <- if (logx) log2(x) else x
    yy <- if (logy) log2(y) else y
    if (is.null(loggood))
        loggood <- !(is.na(xx) | is.infinite(xx) | is.na(yy) | is.infinite(yy))

    say("Pearson correlation r=", report.cor(xx[loggood],yy[loggood],method="pearson"),
        if (logx || logy) " (log transformed values)" else "")
    say("Spearman rank correlation rho=", report.cor(x,y,method="spearman"))

    if (logx) xlab <- sprintf("log2 %s", xlab)
    if (logy) ylab <- sprintf("log2 %s", ylab)
    
    show.plot(name, 
        function()
            plot(xx[loggood],yy[loggood],cex=0.01,
                 xlab=xlab,ylab=ylab),
        width=5, height=5
        )
    
    #show.ggplot(name,
    #    ggplot(data.frame(x=x[good],y=y[good]), aes(x=x,y=y)) + theme 
    #    + coord_trans(x=if (logx) "log10" else "identity",
    #                  y=if (logy) "log10" else "identity")
    #    + geom_point()
    #    )
}









