namespace Tars.Core

module Example =
    let add a b = a + b
    
    let complexFunction x =
        if x > 0 then
            if x > 10 then
                if x > 100 then
                    "Very big"
                else
                    "Big"
            else
                "Small"
        else
            "Non-positive"
