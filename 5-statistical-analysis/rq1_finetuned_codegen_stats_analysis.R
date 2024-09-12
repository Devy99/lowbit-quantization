# Clear the environment
rm(list=ls())

# Download the required libraries if not installed
if (!require("exact2x2")) install.packages("exact2x2")
if (!require("xtable")) install.packages("xtable")
if (!require("effsize")) install.packages("effsize")
if (!require("nortest")) install.packages("nortest")

# Load the required libraries
library(exact2x2)
library(xtable)
library(effsize)
library(nortest)

data<-read.csv("./results-rq1-finetuned/code_generation_results.csv")

# Init results list
res=list(base_precision = c(), quant_precision = c(), lng = c(), model = c(), p.value = c(), OR = c())

languages=c('py', 'java')
models=c('codellama-7b', 'deepseekcoder-7b')
quantizations=c('3bit', '2bit')

for (model in models) {
  for (lng in languages) {
    for (quant in quantizations) {
      quant_finetuned=paste(quant, "-finetuned", sep="")
      baseline_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==quant),]$pass
      quantization_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==quant_finetuned),]$pass

      # McNemar test
      mn=mcnemar.exact(baseline_pass,quantization_pass)
      p.value=mn$p.value
      or=mn$estimate

      # Store the results
      res$base_precision=c(res$base_precision,quant)
      res$quant_precision=c(res$quant_precision,quant_finetuned)
      res$lng=c(res$lng,lng)
      res$model=c(res$model,model)
      res$p.value=c(res$p.value,p.value)
      res$OR=c(res$OR,or)
    }
  }
}

# Generate the dataframes and export to csv
res=data.frame(res)

# Sort dataframe
res=res[order(match(res$model, c('deepseekcoder-7b','codellama-7b')), res$lng, res$base_precision, match(res$quant_precision, c('3bit', '2bit'))),]
write.csv(res, file = "./results-rq1-finetuned/stats_analysis.csv")