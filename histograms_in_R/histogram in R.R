#Plot Air Passengers data
plot(AirPassengers)

#Plot a histogram for Air Passengers data
hist(AirPassengers)

#See how the petal length is distributed
plot(iris$Petal.Length)

#Plot the histogram for iris petal length
hist(iris$Petal.Length)

#Distribution or Species in iris data
plot(iris$Species)

#Try making a histogram for iris species
hist(iris$Species)

#Get the documentation for hist() function
?hist
# ## Default S3 method:
# hist(x, breaks = "Sturges",
#      freq = NULL, probability = !freq,
#      include.lowest = TRUE, right = TRUE,
#      density = NULL, angle = 45, col = NULL, border = NULL,
#      main = paste("Histogram of" , xname),
#      xlim = range(breaks), ylim = NULL,
#      xlab = xname, ylab,
#      axes = TRUE, plot = TRUE, labels = FALSE,
#      nclass = NULL, warn.unused = TRUE, ...)

#Add all the labels
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count")

#Add all the labels and color
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count",border="red", col="blue")

#Add all the labels and color. Set the y axis indexes horizontal
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count",border="red", col="blue",las=1)

#Add all the labels and color. Set the y axis indexes horizontal. Set limits for axis and
#set 6 breaks
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count",border="red", col="blue",las=1,xlim=c(1,7),ylim=c(0,40),breaks=6)

#Add all the labels and color. Set the y axis indexes horizontal. Set limits for axis and
#set probabilistic plot
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count",border="red", col="blue",las=1,xlim=c(1,7),ylim=c(0,1),freq=FALSE)

#Add all the labels and color. Set the y axis indexes horizontal. Set limits for axis and
#Setting color density in lines per inch and the angle
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count",border="red", col="blue",las=1,xlim=c(1,7),ylim=c(0,40),density=50,angle=60)

#Getting the output instead of the plot
hist(iris$Petal.Length,plot=FALSE)

#Add all the labels and color. Set the y axis indexes horizontal. Set limits for axis and
#Drawing lables on top of bars
hist(iris$Petal.Length,main="Histogram for petal length", xlab = "Petal length in cm", ylab = "Count",border="red", col="blue",las=1,xlim=c(1,7),ylim=c(0,40),labels=TRUE)
