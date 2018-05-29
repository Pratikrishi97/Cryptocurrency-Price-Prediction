%{
#include<stdio.h>
int cl_COUNT=0,c=0;
%}
s [//]
d [/*]([.]|[\n])*[*/]
%%
{s} cl_COUNT++;
{d} cl_COUNT++;
. ;
[\n] ;
%%
int main(){
	yylex();
	printf("\ncomment lines:%d\n",cl_COUNT);
	return 1;
}
int yywrap(){
	return 1;
}

