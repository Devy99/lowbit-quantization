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

data<-read.csv("./results-rq2/code_generation_results.csv")

# Init results list
res=list(base_precision = c(), quant_precision = c(), lng = c(), model = c(), p.value = c(), OR = c())

languages=c('py', 'java')
models=c('codellama-7b', 'deepseekcoder-7b')
quantizations=c('8bit', '4bit', '3bit', '2bit')

for (model in models) {
  for (lng in languages) {
    for (quant in quantizations) {
      technique_rnd=paste0(quant, '-rnd')
      technique_mixed=paste0(quant, '-mixed')
      technique_code=paste0(quant, '-code')

      #### Random vs Mixed ####
      baseline_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==technique_rnd),]$pass
      quantization_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==technique_mixed),]$pass

      # McNemar test
      mn=mcnemar.exact(baseline_pass,quantization_pass)
      p.value=mn$p.value
      or=mn$estimate

      # Store the results
      res$base_precision=c(res$base_precision,technique_rnd)
      res$quant_precision=c(res$quant_precision,technique_mixed)
      res$lng=c(res$lng,lng)
      res$model=c(res$model,model)
      res$p.value=c(res$p.value,p.value)
      res$OR=c(res$OR,or)


      #### Random vs Code ####
      baseline_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==technique_rnd),]$pass
      quantization_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==technique_code),]$pass

      # McNemar test
      mn=mcnemar.exact(baseline_pass, quantization_pass)
      p.value=mn$p.value
      or=mn$estimate

      # Store the results
      res$base_precision=c(res$base_precision,technique_rnd)
      res$quant_precision=c(res$quant_precision,technique_code)
      res$lng=c(res$lng,lng)
      res$model=c(res$model,model)
      res$p.value=c(res$p.value,p.value)
      res$OR=c(res$OR,or)


      #### Mixed vs Code ####
      baseline_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==technique_mixed),]$pass
      quantization_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==technique_code),]$pass

      # McNemar test
      mn=mcnemar.exact(baseline_pass, quantization_pass)
      p.value=mn$p.value
      or=mn$estimate

      # Store the results
      res$base_precision=c(res$base_precision,technique_mixed)
      res$quant_precision=c(res$quant_precision,technique_code)
      res$lng=c(res$lng,lng)
      res$model=c(res$model,model)
      res$p.value=c(res$p.value,p.value)
      res$OR=c(res$OR,or)

      # Adjust p-values
      pred_str = paste0("^", quant)
      quant_indices <- which(res$model == model & res$lng == lng & grepl(pred_str, res$base_precision))
      res$p.value[quant_indices]= p.adjust(res$p.value[quant_indices], method = "BH")
    }
  }
}

# Generate the dataframes and export to csv
res=data.frame(res)

# Sort dataframe
res=res[order(match(res$model, c('deepseekcoder-7b','codellama-7b')), res$lng, match(res$base_precision, c('8bit-rnd', '8bit-mixed', '8bit-code', '4bit-rnd', '4bit-mixed', '4bit-code', '3bit-rnd', '3bit-mixed', '3bit-code', '2bit-rnd', '2bit-mixed', '2bit-code'))),]
write.csv(res, file = "./results-rq2/stats_analysis.csv")