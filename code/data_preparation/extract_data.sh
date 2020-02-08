outdir=/scVI/data
mkdir -p $outdir/
mkdir -p $outdir/original

unzip /scVI/scVI-reproducibility/additional/data.zip -d $outdir/original

mkdir -p $outdir/original/CORTEX
wget -v -O $outdir/original/CORTEX/expression_mRNA_17-Aug-2014.txt https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt
