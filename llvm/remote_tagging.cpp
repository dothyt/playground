#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/Analysis/LoopInfo.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include "remote_tagging.hpp"

#define DEBUG_TYPE "remote_tagging"

using namespace llvm;

cl::list<std::string> RType("rtype", cl::desc("Specify remote type name"), cl::OneOrMore);

bool InjectFuncCall::runModule(Module &M, llvm::ModuleAnalysisManager &AM) {
	bool InsertedAtLeastOnePrintf = false;

	auto &CTX = M.getContext();
	PointerType *PrintfArgTy = PointerType::getUnqual(Type::getInt8Ty(CTX));

	

	// STEP 1: Inject the declaration of printf
	// ----------------------------------------
	// Create (or _get_ in cases where it's already available) the following
	// declaration in the IR module:
	//    declare i32 @printf(i8*, ...)
	// It corresponds to the following C declaration:
	//    int printf(char *, ...)
	FunctionType *PrintfTy = FunctionType::get(
			IntegerType::getInt32Ty(CTX),
			PrintfArgTy,
			/*IsVarArgs=*/true);

	FunctionCallee Printf = M.getOrInsertFunction("printf", PrintfTy);

	// Set attributes as per inferLibFuncAttributes in BuildLibCalls.cpp
	Function *PrintfF = dyn_cast<Function>(Printf.getCallee());
	PrintfF->setDoesNotThrow();
	PrintfF->addParamAttr(0, Attribute::NoCapture);
	PrintfF->addParamAttr(0, Attribute::ReadOnly);


	// STEP 2: Inject a global variable that will hold the printf format string
	// ------------------------------------------------------------------------
	llvm::Constant *PrintfFormatStr = llvm::ConstantDataArray::getString(
		CTX, "(llvm-tutor) Hello from: %s\n(llvm-tutor)   number of arguments: %d\n");

	Constant *PrintfFormatStrVar =
		M.getOrInsertGlobal("PrintfFormatStr", PrintfFormatStr->getType());
	dyn_cast<GlobalVariable>(PrintfFormatStrVar)->setInitializer(PrintfFormatStr);

	// STEP 3: For each function in the module, inject a call to printf
	// ----------------------------------------------------------------
	auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
	for (auto &F : M) {
		errs() << "type: " << F.getName() << "\n";
		if (F.isDeclaration())
			continue;

		auto &loopInfo = FAM.getResult<LoopAnalysis>(F);

		
		for (auto loop : loopInfo) {
			auto B = loop->getHeader();
			errs() << "loop: " << loop->getHeader()->getName() << "\n";
			for (Instruction &I: *B) {
				errs() << "inst: " << I << "\n";
			}

			llvm::SmallVector< llvm::BasicBlock * > ExitingBlocks;
			loop->getExitingBlocks(ExitingBlocks);
			for (auto BB : ExitingBlocks) {
				errs() << "Exiting: " << BB->getName() << "\n";
				for (Instruction &I: *BB) {
					errs() << "inst: " << I << "\n";
				}
			}
		}
		
		// LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();

		// Get an IR builder. Sets the insertion point to the top of the function
		IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());

		// Inject a global variable that contains the function name
		auto FuncName = Builder.CreateGlobalStringPtr(F.getName());
		
		// Printf requires i8*, but PrintfFormatStrVar is an array: [n x i8]. Add
		// a cast: [n x i8] -> i8*
		llvm::Value *FormatStrPtr =
			Builder.CreatePointerCast(PrintfFormatStrVar, PrintfArgTy, "formatStr");

		// The following is visible only if you pass -debug on the command line
		// *and* you have an assert build.
		LLVM_DEBUG(dbgs() << " Injecting call to printf inside " << F.getName()
											<< "\n");

		// Finally, inject a call to printf
		Builder.CreateCall(
				Printf, {FormatStrPtr, FuncName, Builder.getInt32(F.arg_size())});

		InsertedAtLeastOnePrintf = true;
	}

	return InsertedAtLeastOnePrintf;
}

bool TypeTracker::structContainsRemoteType(StructType *type) {
	if (typeVisited(type))
		return ifNeedTag(type);

	bool contains_remote = false;
	for (auto &elem : type->elements()) {
		contains_remote |= containsRemoteType(elem);
	}

	if (contains_remote)
		types_to_tag.insert(type);
	visited_types.insert(type);
	return contains_remote;
}


bool TypeTracker::ptrPointToRemoteType(Type *type) {
	// pointer only has one sub element
	auto deref_type = type->getContainedType(0);
	if (deref_type->isStructTy()) {
		return ifNeedTag(cast<StructType>(deref_type));
	} else if (deref_type->isPointerTy()) {
		return ptrPointToRemoteType(deref_type);
	} else if (deref_type->isIntegerTy() || deref_type->isFloatingPointTy() || 
				deref_type->isFunctionTy()) {
		return false;
	} else if (deref_type->isArrayTy()) {
		return containsRemoteType(deref_type);
	} else {
		llvm::report_fatal_error("Unknown type, could be impl error\n");
	}
}

	
/* Rule:
 * 1. remote type will keep track struct that is remote
 * 2. if a struct contains a remote type ptr element, then it needs to be tagged
 * 3. if a struct contains a ptr points to a type needed to be tag,
 *    then it needs to be tagged
 */
bool TypeTracker::containsRemoteType(llvm::Type *type) {
	if (type->isStructTy()) {
		return structContainsRemoteType(cast<StructType>(type));
	} else if (type->isPointerTy()) {
		return ptrPointToRemoteType(type);
	} else if (type->isArrayTy()) {
		return containsRemoteType(cast<ArrayType>(type)->getElementType());
	} else if (type->isIntegerTy() || type->isFloatingPointTy() || 
				type->isFunctionTy()) {
		// do nothing, end point of recursion
		return false;
	} else {
		llvm::report_fatal_error("Unknown type, could be impl error\n");
	}
}

/*
 * we have run a second pass because first pass may miss some members
 * that are not tagged
 */
// void TypeTracker::constructTypeMembersToTag(llvm::Module &M) {
// 	auto module_datalayout = M.getDataLayout();
// 	for (auto type : M.getIdentifiedStructTypes()) {
// 		errs() << "type: " << type->getName() << "\n";
// 		for (auto member_it = type->element_begin(); 
// 				member_it != type->element_end(); member_it++) {
// 			if ((*member_it)->isPointerTy()) {
// 				auto deref_type = (*member_it)->getContainedType(0);
// 				if (deref_type->isStructTy() && ifNeedTag(cast<StructType>(deref_type))) {
// 					unsigned member_idx = std::distance(type->element_begin(), member_it);
// 					auto struct_layout = module_datalayout.getStructLayout(cast<StructType>(type));
// 					auto member_offset = struct_layout->getElementOffset(member_idx);
// 					errs() << "type: " << type->getName() << " member: " << member_idx << " " << member_offset << "\n";
// 					type_members_to_tag.push_back(
// 						std::make_tuple(type, (*member_it)->getTypeID(), 
// 										member_idx, member_offset)
// 					);
// 				}
// 			}
// 		}
// 	}
// }


void TypeTracker::runModule(llvm::Module &M) {
	auto &CTX = M.getContext();
	auto list_type = M.getIdentifiedStructTypes();

	if (RType.empty()) {
		llvm:report_fatal_error("No remote type specified\n" 
			"\tuse -mllvm -rtype=<type name> to specify remote type\n", false);
	}

	// set up the tagged types
	for(auto arg: RType) {
		for (auto &type : list_type) {
			if (type->getName().equals(arg)) {
				types_to_tag.insert(type);
			}
		}
	}

	for(auto arg: RType) {
		errs() << "arg: " << arg << "\n";
	}

	// find alll the types related to remote type given
	for (auto &type : list_type) {
		structContainsRemoteType(type);
	}

	for (auto it = types_to_tag.begin(); it != types_to_tag.end(); it++) {
		errs() << "types_to_tag: " << (*it)->getName() << " " << *it << "\n";
	}

	// constructTypeMembersToTag(M);
}

struct LoopChecker : public PassInfoMixin<LoopChecker> {
	PreservedAnalyses run(Loop &L, LoopAnalysisManager &LAM, 
						  LoopStandardAnalysisResults &LSAR,
						  LPMUpdater &Updater) {
		errs() << "LoopChecker\n";
		return PreservedAnalyses::all();
	}
};


PassPluginLibraryInfo getInjectFuncCallPluginInfo() {
	const auto callback = [](PassBuilder &PB) {
		PB.registerPipelineStartEPCallback(
			[&](ModulePassManager &MPM, auto) {
				MPM.addPass(TypeTracker());
				MPM.addPass(InjectFuncCall());
				return true;
			}
		);

		// PB.registerPipelineParsingCallback(
		// 	[](StringRef Name, ModulePassManager &MPM,
        //            ArrayRef<PassBuilder::PipelineElement>) {
		// 		MPM.addPass(TypeTracker());
		// 		MPM.addPass(InjectFuncCall());
		// 		return true;
		// 	}
		// );

		PB.registerLoopOptimizerEndEPCallback(
			[&](LoopPassManager &LPM, auto) {
				LPM.addPass(LoopChecker());
				return true;
			}		
		);
		
	};

	return {LLVM_PLUGIN_API_VERSION, "remote_tagging", "0.0.1", callback};
};


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
	return getInjectFuncCallPluginInfo();
}