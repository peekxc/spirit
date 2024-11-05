
#let funcs = (
	"sgn","Im","Ker","dim","dgm","up","dn","Hom","rank","diag",
	"Tr","qr","tr","Pr","col","card","ker","nullity","sign","deg",
	"co","low","R","Rips","B","diam","SW",
)
#let (sgn,Im,Ker,dim,dgm,up,dn,Hom,rank,diag,Tr,qr,tr,Pr,col,card,ker,nullity,sign,deg,co,low,R,Rips,B,diam, SW) = funcs.map(math.op)

#let argmax = math.op("arg max", limits: true)
#let argmin = math.op("arg min", limits: true)