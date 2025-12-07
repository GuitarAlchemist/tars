let reverseList lst =
	let rec helper acc lst =
		match lst with
		| [] -> acc
		| head :: tail -> helper (head :: acc) tail
	in
	helper [] lst