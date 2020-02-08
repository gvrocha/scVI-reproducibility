library(Matrix)
library(scone)
library(plyr) 

originals_dir = "/scVI/data/original/PBMC"
processed_dir = "/scVI/data/original/PBMC"

# get cleaned data by merging genes from original scone pipeline and microarray
load(file.path(originals_dir, "scVI_scone.rda"))

barcodes = scone_obj@colData@rownames
list_qc = scone_obj@colData@listData[names(scone_obj@colData@listData)[1:9]]
qc.df = do.call("cbind", lapply(list_qc, as.data.frame))
colnames(qc.df) = names(scone_obj@colData@listData)[1:9]
batch = get_batch(scone_obj)
gene_names = scone_obj@NAMES
design = get_design(scone_obj, method = "none,fq,qc_k=8,no_bio,no_batch" )

write.csv(barcodes,   file.path(processed_dir, "barcodes.csv"))
write.csv(batch,      file.path(processed_dir, "batch.csv"))
write.csv(qc.df,      file.path(processed_dir, "full_qc.csv"))
write.csv(gene_names, file.path(processed_dir, "gene_names.csv"))
write.csv(design,     file.path(processed_dir, "design.csv"))

# load cells information from SEURAT, included in the original scone object
load(file.path(originals_dir, "scone_all_wposcon_extendim_biostrat2_q.rda"))
bio = get_bio(scone_obj)
write.csv(bio, file.path(processed_dir, "bio.csv"))
