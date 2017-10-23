function disp( str )
%DISP disp function override
    import nnf.utl.Globals;
    if (Globals.DEBUG_PRINT)
        disp(str);
    end
end

