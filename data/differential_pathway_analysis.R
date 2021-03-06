#
# Pathway activity based discriminative analysis
#


#
# Convert gene expression matrix to summarized pathway activity matrix using different dimension reduction or averaging approaches
#
summarized_pathway_activity = function(exprs, gsets=NULL, type="median", minsize = 10)
{
  
  if(is.null(gsets) && is.null(database))
  {
    stop("Either the gsets or the database parameter must be specified!")
  }
  
  exprs <- as.matrix(exprs)

	genenames = rownames(exprs)
	
	# list of pathway expression matrices
	pathmat = matrix(0, nrow=length(gsets), ncol=ncol(exprs))
	rownames(pathmat) = rep("", nrow(pathmat))

	count = 0	
	cat('\n',length(gsets),' pathways read.\n')
	for(j in 1:length(gsets))
	{
	  if(j %% 100 == 0){
	  	cat(paste("Current iteration:",j,"\n"))
	  }
	  
	  gset = gsets[[j]]
	  
	  mapid = match(gset, genenames)
	  
	  notna = which(!is.na(mapid))
	  
	  if(length(notna) <  minsize)
	  	next
	
	  curpathmat = exprs[mapid[notna],]
	  
	  meanpathvec = NULL
	  if(type == "mean") {
	  	meanpathvec = apply(curpathmat, 2, mean)
	  } else if(type=="min"){
	    meanpathvec = apply(curpathmat, 2, min)
	  } else if(type=="max"){
	    meanpathvec = apply(curpathmat, 2, max)	
	  } else if(type=="sd"){
	    meanpathvec = apply(curpathmat, 2, sd)
	  } else if(type=="pca"){
	    rem = which(apply(curpathmat, 1, var)==0)
	    curpathmatfilt = curpathmat
	    if(length(rem))
	    	curpathmatfilt = curpathmat[-rem,]
	    if(length(curpathmatfilt))
	    {
	    	pca    <- prcomp(t(curpathmatfilt), retx=T, scale=T) # scaled pca 
				scores <- pca$x[,1]
				meanpathvec = scores
			} else {
				meanpathvec = rep(0, ncol(exprs))
			}
		} else if(type=="mds") {
		  meanpathvec <- as.vector(cmdscale(dist(t(curpathmat)), k = 1))
	  } else {
	  	meanpathvec = apply(curpathmat, 2, median)
	  } 	  
	  	  
	  count = count + 1
	  pathmat[count,] = meanpathvec
	  rownames(pathmat)[count] = names(gsets)[j]
	}
	
	pathmat = pathmat[1:count,]

  return(pathmat)
}


#
# Load pathway data from the MSIGDB database
#
load_pathway_database = function(database = c("CANONIC","GO","KEGG","REACTOME","PID","BIOCARTA","CHRPOSITION"), vnum="7.4")
{

	msigdb_pathways = NULL
	if(database=="CANONIC")
		msigdb_pathways = sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c2.cp.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])
	if(database=="GO")
		msigdb_pathways = sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c5.all.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])
	if(database=="KEGG")
		msigdb_pathways = sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c2.cp.kegg.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])
	if(database=="REACTOME")
		msigdb_pathways =sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c2.cp.reactome.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])
	if(database=="PID")
		msigdb_pathways = sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c2.cp.pid.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])
	if(database=="BIOCARTA")
		msigdb_pathways = sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c2.cp.biocarta.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])
	if(database=="CHRPOSITION")
		msigdb_pathways = sapply(readLines(paste("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",vnum,"/c1.all.v",vnum,".symbols.gmt",sep="")), function(x) strsplit(x, "\t")[[1]])

	return(msigdb_pathways)
}

#
# Run an example analysis on Parkinson's disease data from the GEO database (requires Affymetrix annotation data in file: HG-U133A.na36.annot.csv)
#
run_example = function()
{

		#
		# Load pathway data
		#
		
		pathdat = load_pathway_database("KEGG")

		# convert to required list format
		pathlst = sapply(pathdat, function(x) x[3:length(x)])
		
		names(pathlst) = sapply(pathdat, function(x) x[1])

		
		#
	  # Load example Parkinson's disease case/control gene expression dataset from GEO
	  #
	  #    Dataset GSE8397: L. B. Moran et al., Neurogenetics, 2006, SN + frontal gyrus, post mortem,	PD (29), healthy (18)
		#    Array platform: Affymetrix HG-U133A
		#

		# Download the data into the current working directory

		# for Windows - manually via the web-browser using this url:
		#
		# ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE8nnn/GSE8397/matrix/GSE8397-GPL96_series_matrix.txt.gz
		#

		# for Mac/Linux - automatically via R command line
		system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE8nnn/GSE8397/matrix/GSE8397-GPL96_series_matrix.txt.gz')

		# Read the data into R
		morandatgeo = read.table(gzfile("GSE8397-GPL96_series_matrix.txt.gz"), header=T, comment.char="!", sep="\t")

		# Use the labels in the first column as row names
		morandat = morandatgeo[,2:ncol(morandatgeo)]
		rownames(morandat) = morandatgeo[,1]
		
		# Filter out tissue samples which are not from the midbrain / substantia nigra region
		moran_tissues = as.matrix(read.table(gzfile("GSE8397-GPL96_series_matrix.txt.gz"), header=F, nrows=1, skip=36, sep="\t"))
		moran_tissues = moran_tissues[2:length(moran_tissues)]
		
		nigra_ind = grep("substantia nigra",moran_tissues)
		
		moran_outcome = as.matrix(read.table(gzfile("GSE8397-GPL96_series_matrix.txt.gz"), header=F, nrows=1, skip=28, sep="\t"))
		moran_outcome = moran_outcome[2:length(moran_outcome)]
		moran_outcome[grep("control",moran_outcome)] = rep("control",length(grep("control",moran_outcome)))
		moran_outcome[grep("Parkinson",moran_outcome)] = rep("parkinson",length(grep("Parkinson",moran_outcome)))
		
		moranfilt = morandat[,nigra_ind]
		#dim(moranfilt)
		#[1] 22283    39
		
		moran_outcomefilt = moran_outcome[nigra_ind]
		#table(moran_outcomefilt)
		#moran_outcomefilt
		#  control Parkinson 
		#       15        24

		# Unzip annotation file (in Mac/Linux, needs to be done manually in Windows)
		system('unzip HG-U133A.na36.annot.csv.zip')
		
		# read annotations file (ignoring comments)
		annot = read.csv("HG-U133A.na36.annot.csv", comment.char="#")
		head(annot)
		
		# map probes to microarray rownames
		mapids = match(rownames(moranfilt), annot$Probe.Set.ID)
		
		# check if all IDs were mapped successfully
		any(is.na(mapids))
		#[1] FALSE
		# ok, no missing IDs
		
		# extract gene symbols corresponding to microarray Probe IDs (take always the first symbol mapped)
		mapped_symbols = sapply( as.character(annot$Gene.Symbol[mapids]) , function(x) strsplit(x, " /// ")[[1]][1])
		

			
		#
		# Convert expression matrix with Affymetrix IDs to Gene Symbol matrix (if multiple probes match to a gene, take the max. average value probe as representative for the gene)
		#
		
		# Function to convert probe-based expression matrix to gene-based expression matrix
		# Parameters:
		#   matdat = input matrix with probe rownames,
		#   mat_conv = vector with gene symbols corresponding to probe rownames (NA for missing conversions)
		probe2genemat <- function(matdat, mat_conv)
		{
		
			if(nrow(matdat) != length(mat_conv))
			{
			  stop("Matrix does not have the same number of rows as the gene name vector")
			}
		
			# take max expression vector (max = maximum of mean exp across samples), if gene occurs twice among probes
			unq <- unique(mat_conv)
			if(any(is.na(unq))){
				unq <- unq[-which(is.na(unq))]
			}
			mat <- matrix(0.0, nrow=length(unq), ncol=ncol(matdat))
			for(j in 1:nrow(mat))
			{
			  ind <- which(unq[j]==mat_conv)
		
			  # show conversion progress, every 1000 probes
			  if(j %% 1000 == 0){
			    print(j)
			  }
		
			  # 1-to-1 probe to gene symbol matching
			  if(length(ind) == 1)
			  {
			    mat[j,] = as.numeric(as.matrix(matdat[ind,]))
			  } else if(length(ind) > 1){
		
			    # multiple probes match to one gene symbol
			    curmat = matdat[ind,]
		
			    # compute average expression per row -> select row with max. avg. expression
			    avg = apply(curmat, 1, mean)
			    mat[j,] = as.numeric(as.matrix(matdat[ind[which.max(avg)],]))
			  }
			}
			rownames(mat) = unq
		
		  return(mat)
		}
		
		# Run the conversion from probe matrix to gene matrix (Moran data)
		moran_symb = probe2genemat(moranfilt, mapped_symbols)
		colnames(moran_symb) = colnames(moranfilt)		
		# dim(moran_symb)

	  
	  # Extract pathway activities
		moran_path = summarized_pathway_activity(moran_symb, gsets=pathlst, type="median", minsize = 10)			
		print(moran_path[1:5,1:5])


		# Random Forest classification analysis		
		
		# install R-packages for classification
		if(!require('randomForest'))
		{
			install.packages('randomForest')
			require('randomForest')
		}
		
		
		# set seed number for reproducibility
		set.seed(1234) 
		
		# Build Random Forest sample classification model for Zhang et al. data using 250 decision trees		
		rfmod_moran= randomForest(t(moran_path), factor(moran_outcomefilt), ntree=250, keep.forest=TRUE)
		
		# show model evluation based on out-of-bag samples
		print(rfmod_moran)
		# OOB estimate of  error rate: 10.26%
		
		
		
		# which variables were most informative for the prediction (multivariate feature selction):
		pathways_pd_vs_control_ranked = rfmod_moran$importance[order(rfmod_moran$importance, decreasing=T),]
		
		print(head(pathways_pd_vs_control_ranked))
	
		
		moran_ages = as.matrix(read.table(gzfile("GSE8397-GPL96_series_matrix.txt.gz"), header=F, nrows=1, skip=38, sep="\t"))
		moran_ages = as.numeric(sapply(moran_ages, function(x) strsplit(x," |;")[[1]][2]))
		moran_ages = moran_ages[2:length(moran_ages)]
	
	
		# set seed number for reproducibility
		set.seed(1234) 
		
		# Build Random Forest sample classification model for Zhang et al. data using 250 decision trees		
		rfmod_moran= randomForest(t(moran_path), moran_ages[nigra_ind], ntree=250, keep.forest=TRUE)
		# Mean of squared residuals: 74.42821
		#% Var explained: 24.1
		#cor(rfmod_moran$predicted, moran_ages[nigra_ind])
		# 0.4962768

		# Correlation plot: Predicted vs. real chronological age		
		linreg <- lm(rfmod_moran$predicted ~ moran_ages[nigra_ind])
		plot(moran_ages[nigra_ind], rfmod_moran$predicted, xlab="real chronological age", ylab="predicted age")
		abline(linreg, col="red", lwd=1.2)
		
		# Statistics
		print(summary(linreg))
		# p-value: 0.005113

		
		# top-ranked pathways		
		pathways_agepred_ranked = rfmod_moran$importance[order(rfmod_moran$importance, decreasing=T),]
		print(head(pathways_agepred_ranked))
		
	
		# Heat map visualization of top 10 pathways	
		require('gplots')
		
		curdat = as.matrix(moran_path[match(names(pathways_agepred_ranked)[1:10], rownames(moran_path)),])
		
		heatmap.2(curdat, hclustfun=function(dist) {hclust(dist, method='average')}, distfun=function(x) {as.dist(1-cor(t(x), method="pearson"))}, key=TRUE, symkey=FALSE, col=colorpanel(511, "blue","white","red"),density.info="none", trace="none", cexRow=0.5, dendrogram="both", scale="row")
		
	
	
		write.table(moran_ages[nigra_ind], file="moran_ages.txt", quote=F, col.names=F, row.names=F)
		write.table(moran_outcomefilt[nigra_ind], file="moran_clinical_outcome.txt", quote=F, col.names=F, row.names=F)
		write.table(round(moran_path,2), file="moran_pathway_activity_matrix.txt", quote=F, col.names=F, row.names=T)
		
		
		
		#
		# Load data for GSE25219 dataset (Human Brain Transcriptome Atlas)
		#
		
		
		# gene expression data
		brainall_data <- read.table(gzfile("GSE25219-GPL5175_series_matrix.txt.gz"), header=T, comment.char="!", sep="\t")
		# use identifiers in first column as rownames
		brainall = as.matrix(brainall_data[,2:ncol(brainall_data)])
		rownames(brainall) = brainall_data[,1]
		
		# annotation
		annotmat = read.table("GSE25219_HBTAtlas_annotation_data.txt", header=TRUE, sep="\t")
		sample_all = as.character(annotmat[,1])
		stageall = annotmat[,2]
		gendersall = ifelse(annotmat[,3]=="f",0,1)
		brainregions = sapply(annotmat[,1], function(x) strsplit(as.matrix(x), "_")[[1]][2])		
		
		annotfull = read.table('GSE25219_GPL5175_full_annotations.txt', header=FALSE, sep="\t", comment.char="#", fill=T)
		#all(match(annotfull$V1, colnames(brainall)) == 1:ncol(brainall))
		#[1] TRUE
				
		# load gene labels (manually extracted from GPL platform annotations)
		genelabmat = read.table("GSE25219_HBTAtlas_gene_symbols.txt", header=TRUE, sep="\t")
		genelab = as.character(genelabmat[,2])
		
		rownames(brainall) = genelab

		# focus only on samples derived from adults above age 30 (stage >= 13)
		# stage 13 = 20-40 years
		# stage 14 = 40-60 years
		# stage 15 = 60+ years
		adultsamp = which(stageall >= 13) # 404 samples
		
		
		# focus only on midbrain samples (striatum = STR)
		table(brainregions)
		#brainregions
		#A1C AMY CBC CGE DFC DIE DTH  FC HIP IPC ITC LGE M1C  MD MFC MGE MSC  OC OFC  PC S1C STC STR  TC URL 
		# 83  76  71   4  88   3   4   3  82  85  78   4  72  73  90   4  13   5  82   6  72  86  75   5   6 
		#V1C  VF VFC 
		# 83   2  85 
		
		striatum = which(brainregions == "STR")
		
		striatum_adult = intersect(adultsamp, striatum)
		# 26 samples 
		
		#table(annotfull$V8[striatum_adult])[which(table(annotfull$V8[striatum_adult])!=0)]
		#21 Y 22 Y 23 Y 27 Y 30 Y 36 Y 37 Y 40 Y 42 Y 55 Y 64 Y 70 Y 82 Y 
   		#	1    2    2    1    2    2    4    4    1    1    2    2    2
   	
   		striatum_adult_ages = as.numeric(sapply(striatum_adult_ages, function(x) strsplit(as.matrix(x), " ")[[1]][1]))
				
		brain_path = summarized_pathway_activity(brainall[,striatum_adult], gsets=pathlst, type="median", minsize = 10)			
				
		write.table(striatum_adult_ages, file="human_brain_transcriptome_atlas_ages.txt", quote=F, col.names=F, row.names=F)
		write.table(round(brain_path,2), file="human_brain_transcriptome_atlas_pathway_activity_matrix.txt", quote=F, col.names=F, row.names=T)
		
		
}

