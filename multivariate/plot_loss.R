list.of.packages <- c("data.table", "ggplot2", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

setwd("~/git/bse-transformer-seminar/multivariate/")

dat = read.delim("loss.txt", sep=" ", header=F)
dat = dat[,c("V2", "V3", "V5")]
dat$V2 = as.numeric(gsub("]", "", dat$V2))
names(dat) = c("epoch", "set", "loss")

ggplot(dat, aes(x=epoch, y=loss, group=set, color=set)) + geom_line()
