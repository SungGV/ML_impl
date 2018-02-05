package kmean;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class Kmeans {

	private final double TOLERANCE=.01;

	public static void main(String[] args){
		Kmeans km=new Kmeans();
		double[][] inst=km.read("/Users/Sung/Desktop/school/335/hw09data.csv");
//		double[][] inst=km.read("/Users/Sung/Desktop/school/335/proj02data.csv");
		
		int[] c=km.cluster(inst,4);
		for(int i=0; i<inst.length; i++)
			System.out.println(i+"\t"+c[i]);
		

		km.normalization(inst);

		System.out.println("------------------------------");

		for(int i=0; i<inst.length; i++) {
			for(int z = 0; z < inst[i].length; z++) {
				System.out.println(inst[i][z]);
			}
		}
		int[] c2 =km.cluster(inst,4);
		for(int i=0; i<inst.length; i++)
			System.out.println(i+"\t"+c2[i]);

		km.outliers(inst, km.init(inst, 4), c, 4);
	} 

	public int[] cluster(double[][] inst, int k){
		int[] clusters=new int[inst.length];
		double[][] centroids=init(inst,k);
		double errThis=sse(inst,centroids,clusters);
		double errLast=errThis+1;

		while(errLast-errThis>TOLERANCE){
			//reassign the clusters using assignClusters
			clusters = assignClusters(inst, centroids);
			//re-calculate the centroids
			centroids = recalcCentroids(inst, clusters, k);
			//re-calculate the error using sse
			errLast = errThis;
			errThis = sse(inst,centroids,clusters);

		}
		return clusters;
	}

	//finds initial clusters - no modifications necessary
	public double[][] init(double[][] inst, int k){
		int n=inst.length, d=inst[0].length;
		double[][] centroids=new double[k][d];
		double[][] extremes=new double[d][2];
		for(int i=0; i<d; i++)
			extremes[i][1]=Double.MAX_VALUE;
		for(int i=0; i<n; i++)
			for(int j=0; j<d; j++){
				extremes[j][0]=Math.max(extremes[j][0],inst[i][j]);
				extremes[j][1]=Math.min(extremes[j][1],inst[i][j]);
			}
		for(int i=0; i<k; i++)
			for(int j=0; j<d; j++)
				centroids[i][j]=Math.random()*(extremes[j][0]-extremes[j][1])+extremes[j][1];
		return centroids;
	}

	public int[] assignClusters(double[][] inst, double[][] centroids){
		int n=inst.length, d=inst[0].length, k=centroids.length;
		int[] rtn=new int[n];
		double min = 0;
		int index = 0 ;

		//for each instance
		for(int i=0; i<inst.length; i++) {
			min = euclid(inst[i], centroids[0]);
			for(int z = 0; z < centroids.length; z++) {
				if(min >= euclid(inst[i], centroids[z])) {
					min = euclid(inst[i], centroids[z]);
					index = z;
				}
			}
			rtn[i] = index;
		}

		//calculate the distance to each of the different centroids
		//and assign it to the cluster with the lowest distance
		return rtn;
	}

	public double[][] recalcCentroids(double[][] inst, int[] clusters, int k){
		int n=inst.length, d=inst[0].length;
		double[][] centroids=new double[k][d];
		int[] cnt=new int[k];

		//use cnt to count the number of instances in each cluster
		for(int i = 0; i < clusters.length; i++) {
			for(int j = 0; j < cnt.length; j++) {
				if(clusters[i] == j) {
					cnt[j] += 1;
				}
			}
		}
		//for each cluster
		//for each attribute in this cluster
		//add the value of the attribute from each instance in the cluster
		//calculate the averages by dividing each attribute total by the count

		double[][] temp = new double[k][d];


		for(int i=0; i<inst.length; i++) {
			for(int j = 0; j < inst[i].length; j++) {
				temp[clusters[i]][j] += inst[i][j];
			}
		}	

		for(int i=0; i<centroids.length; i++) {
			for(int z=0; z<centroids[i].length; z++) {
				if(cnt[i] != 0) 
					centroids[i][z] = temp[i][z] / cnt[i];
				else 
					centroids[i][z] = 0;
			}
		}

		//do this for each centroid, each attribute
		//be careful not to divide by zero - if a cluster is emply, skip it
		return centroids;
	}

	public double sse(double[][] inst, double[][] centroids, int[] clusters){
		int n=inst.length, d=inst[0].length, k=centroids.length;
		double sum=0;
		//iterate through all instances
		//iterate through all clusters
		//if an instance is in the current cluster, add the euclidean distance
		//between them to the sum
		for(int i=0; i<inst.length; i++) {
			for(int z = 0; z < clusters.length; z++) {
				if(clusters[i] == z ) {
					sum += 	euclid(inst[i],  centroids[z]);
				}
			}
		}
		return sum;
	}

	private double euclid(double[] inst1, double[] inst2){
		double sum=0;
		//calculate the euclidean distance between inst1 and inst2
		for(int i = 0; i< inst1.length; i++) {
			sum += Math.pow(inst1[i] - inst2[i], 2);
		}
		return Math.sqrt(sum);
	}

	//prints out a matrix - can be used for debugging - no modifications necessary
	public void printMatrix(double[][] mat){
		for(int i=0; i<mat.length; i++){
			for(int j=0; j<mat[i].length; j++)
				System.out.print(mat[i][j]+"\t");
			System.out.println();
		}
	}

	//reads in the file - no modifications necessary
	public double[][] read(String filename){
		double[][] rtn=null;
		try{
			BufferedReader br=new BufferedReader(new FileReader(filename));
			ArrayList<String> lst=new ArrayList<String>();
			br.readLine();//skip first line of file - headers
			String line="";
			while((line=br.readLine())!=null)
				lst.add(line);
			int n=lst.size(), d=lst.get(0).split(",").length;
			rtn=new double[n][d];
			for(int i=0; i<n; i++){
				String[] parts=lst.get(i).split(",");
				for(int j=0; j<d; j++)
					rtn[i][j]=Double.parseDouble(parts[j]);
			}
			br.close();
		}catch(IOException e){System.out.println(e.toString());}
		return rtn;
	}

	public void normalization(double[][] inst) {

		double[][] maxMin = new double[2][inst[0].length]; // 5 col, first row -> max, second row -> min
		
		// transposing the table
		// find min and max instance
		for(int j = 0 ; j < inst[0].length; j++) {  // 5
			maxMin[0][j]  = inst[0][j]; // max
			maxMin[1][j]  = inst[0][j];  // min
			for(int i = 0 ; i < inst.length; i++) {   // 100
				if(maxMin[0][j]  <= inst[i][j]) {
					maxMin[0][j]  = inst[i][j];
				}
				if(maxMin[1][j]  >= inst[i][j]) {
					maxMin[1][j]  = inst[i][j];
				}
			}
		}

		// normalization
		for(int i = 0 ; i < inst.length; i++) {
			for(int j = 0 ; j < inst[i].length; j++) {
				inst[i][j] = (inst[i][j] - maxMin[1][j]) / (maxMin[0][j] - maxMin[1][j]);
			}
		}


	}

	public void outliers(double[][] inst, double[][] centroid, int[] cluster, int k) {
		
		double[][] temp = new double[k][inst.length];
		
		// creates temp table with index of instance for each cluster
		// row: cluster
		// col : instance index + 1
		for(int i=0; i<inst.length; i++) {
			for(int j = 0; j < inst[i].length; j++) {
				temp[cluster[i]][i] = i+1;
			}
		}	


		// find the max distance and instance number for each cluster
		double[][] max = new double[2][temp.length];  // row[0] : distance. row[1] : index
		
		for(int i=0; i<temp.length; i++) {
			for(int j=0; j<temp[i].length; j++) {
				if( (int)temp[i][j] > 0) {
					double euclid = euclid(centroid[i], inst[(int)temp[i][j]-1]);
					if(max[0][i] <= euclid) {
						max[0][i] = euclid;
						max[1][i] = (int)temp[i][j]-1;
					}
				}
			}
		}
		// print distance and instance 
		for(int i=0; i<max.length; i++) {
			for(int j=0; j<max[i].length; j++) {
				System.out.println(max[i][j]);
				
			}
		}
	}
}